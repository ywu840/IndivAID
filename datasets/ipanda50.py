import glob
import re

import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
import pickle

class IPanda50(BaseImageDataset):
    """
    IPanda50
    Reference:
        Wang et al. Giant Panda Identification. IEEE Transactions on Image Processing 2021.
    URL: 
        https://doi.org/10.1109/TIP.2021.3055627

    Dataset statistics:
    # identities: 40 (train) + 10 (gallery) + 10 (query)
    # images: 5620 (train) + 677 (gallery) + 577 (query)
    """
    dataset_dir = "IPanda50"

    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(IPanda50, self).__init__()
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
            print("=> IPanda50 loaded")
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
                assert 0 <= pid2label[pid] <= 39
                
            elif set_name == "gallery" or set_name == "query":
                assert 0 <= pid2label[pid] <= 9
            
            
            if relabel:
                pid = pid2label[pid]
                
            dataset.append((img_path, self.pid_begin + pid, camid, 0))

        return dataset