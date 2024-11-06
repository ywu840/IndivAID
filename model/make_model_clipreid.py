from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts): 
        x = prompts + self.positional_embedding.type(self.dtype) 
        x = x.permute(1, 0, 2)  # NLD -> LND 
        x = self.transformer(x) 
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) 

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection 
        return x

class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE   

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        dataset_name = cfg.DATASETS.NAMES
        self.prompt_learner = PromptLearner(num_classes, dataset_name, clip_model)
        self.text_encoder = TextEncoder(clip_model)

    def forward(self, x = None, label=None, get_image = False, get_text = False, cam_label= None, view_label=None, img_features=None):
        if get_text == True and get_image == True:
            prompts = self.prompt_learner(label, img_features)
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            return text_features
        
        if get_text == True:
            prompts = self.prompt_learner(label)
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            return text_features
        
        if get_image == True:
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            if self.model_name == 'RN50':
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                return image_features_proj[:,0]
        
        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1) 
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1) 
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if cam_label != None and view_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label != None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed) 
            img_feature_last = image_features_last[:,0]
            img_feature = image_features[:,0]
            img_feature_proj = image_features_proj[:,0]

        self.bottleneck.to("cuda")
        feat = self.bottleneck(img_feature)
        self.bottleneck_proj.to("cuda")
        feat_proj = self.bottleneck_proj(img_feature_proj) 
        
        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj

        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)


    def load_param(self, trained_path):
        extension = trained_path.split(".")[-1]
        if extension == "pth":
            param_dict = torch.load(trained_path)
            print(param_dict)
            for i in param_dict:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        
        else:
            param_dict = torch.jit.load(trained_path)
            print(param_dict)
            
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model

class PromptLearner(nn.Module):
    def __init__(self, num_id, dataset_name, clip_model):
        super().__init__()
        self.n_id = num_id
        self.n_ctx = 4
        n_cls = 1  # fix the number of class to be 1 in ReID

        # ctx_dim = 512
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim

        if dataset_name == "atrw":
            ctx_init = "A photo of a X X X X tiger."
            
        elif dataset_name == "stoat":
            ctx_init = "A photo of a X X X X stoat."
        
        elif dataset_name == "friesiancattle2017":
            ctx_init = "A photo of a X X X X cattle."
            
        elif dataset_name == "lion":
            ctx_init = "A photo of a X X X X lion."
            
        elif dataset_name == "mpdd":
            ctx_init = "A photo of a X X X X dog."
            
        elif dataset_name == "ipanda50":
            ctx_init = "A photo of a X X X X panda."
            
        elif dataset_name == "seastar":
            ctx_init = "A photo of a X X X X seastar."
            
        elif dataset_name == "nyala":
            ctx_init = "A photo of a X X X X nyala."
           
        elif dataset_name == "polarbear":
            ctx_init = "A photo of a X X X X polarbear."    
        
        
        n_ctx_prefix = 4  # "A photo of a"

        # Use given words to initialize context vectors
        ctx_vectors = torch.empty(self.n_id, self.n_ctx, ctx_dim, dtype=clip_model.dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        
        print("\nInitial Context: {}".format(ctx_init))
        print("Number of Context Tokens: {}\n".format(self.n_ctx))
        self.ctx = nn.Parameter(ctx_vectors)  # To be optimized
        
        self.meta_net = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("linear2", nn.Linear(vis_dim // 16, ctx_dim)),
                ]
            )
        )
        
        ctx_init = ctx_init.replace("_", " ")
        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        with torch.no_grad():
            prompts_embedding = clip_model.token_embedding(self.tokenized_prompts).type(
                clip_model.dtype
            )
            #print(prompts_embedding.shape)
        

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", prompts_embedding[:, : n_ctx_prefix + 1, :])  # SOS (1, prefix_length, ctx_dim)  (1, 5, 512)
        self.register_buffer("token_suffix", prompts_embedding[:, n_ctx_prefix + 1 + self.n_ctx :, :])  #EOS (1, suffix_length, ctx_dim)  (1, 68, 512)
        
        self.dtype = clip_model.dtype

    def forward(self, label, image_features):
        b = label.shape[0]
        ctx = self.ctx[label]  # (batch_size, n_ctx, ctx_dim)
        #print(f"ctx -- {ctx.shape}")
        bias = self.meta_net(image_features)  # (batch_size, ctx_dim)
        #print(f"bias -- {bias.shape}")
        bias = bias.unsqueeze(1)  # (batch_size, 1, ctx_dim)
        #print(f"bias -- {bias.shape}")
        ctx_shifted = ctx + bias  # (batch_size, n_ctx, ctx_dim)  (1, 4, 512)
        #print(f"The shape of the ctx_shifted: {ctx_shifted.shape}\n")

        prefix = self.token_prefix.expand(b, -1, -1)  # (batch_size, prefix_length, ctx_dim)  (1, 5, 512)
        suffix = self.token_suffix.expand(b, -1, -1)  # (batch_size, suffix_length, ctx_dim)  (1, 68, 512)
        

        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0)  # (1, n_ctx, ctx_dim)  (1, 4, 512)
            #ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_id, -1, -1)  # (n_id, n_ctx, ctx_dim)  (*, 4, 512)
            #print(f"The shape of the prefix: {prefix.shape}")
            #print(f"The shape of the ctx_i: {ctx_shifted_i.shape}")
            #print(f"The shape of the suffix: {suffix.shape}\n")

            prompts_i = torch.cat([prefix, ctx_i, suffix], dim=1)  # (n_id, fixed_text_length, ctx_dim)  (*, 77, 512)
            prompts.append(prompts_i)
        
        prompts = torch.stack(prompts)
        prompts = prompts.squeeze(axis = 0)

        return prompts 

