import glob
import re

import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
import pickle

class FriesianCattle2017(BaseImageDataset):
    """
    FriesianCattle2017
    Reference:
        Andrew et al. Visual Localisation and Individual Identification of Holstein Friesian Cattle via Deep Learning. ICCVW 2017.
    URL: 
        https://doi.org/10.1109/ICCVW.2017.336
        or (https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w41/Andrew_Visual_Localisation_and_ICCV_2017_paper.pdf)

    Dataset statistics:
    # identities: 66 (train) + 18 (gallery) + 18 (query)
    # images: 752 (train) + 97 (gallery) + 85 (query)
    """
    dataset_dir = "FriesianCattle2017"

    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(FriesianCattle2017, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)
        

        if verbose:
            print("=> FriesianCattle2017 loaded")
            self.print_dataset_statistics(train, query, gallery)


        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        set_name = dir_path.split("/")[-1]
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))  # id_-1_no_origin.jpg
        
        pattern = re.compile(r'\d+_-?\d+_\d+_?.*')

        pid_container = set()
        camid_container = set()
        
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).group().split("_")[0:2])
            pid_container.add(pid)
            camid_container.add(camid)

        pid2label = {pid : label for label, pid in enumerate(pid_container)}
        assert len(camid_container) == 1

        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).group().split("_")[0:2])
            
            if set_name == "train":
                assert 0 <= pid2label[pid] <= 65
                
            elif set_name == "gallery" or set_name == "query":
                assert 0 <= pid2label[pid] <= 17
            
            
            if relabel:
                pid = pid2label[pid]
                
            dataset.append((img_path, self.pid_begin + pid, camid, 0))

        return dataset