import glob
import cv2
import numpy as np
from torch.utils.data import Dataset

class RainDataset(Dataset):
    def __init__(self, opt):
        super(RainDataset, self).__init__()
        self.dataset=opt
        self.img_list = sorted(glob.glob(self.dataset+'/data/*'))
        self.gt_list = sorted(glob.glob(self.dataset+'/gt/*'))
   
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        gt_name = self.gt_list[idx]

        img = cv2.imread(img_name,cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(gt_name,cv2.IMREAD_COLOR)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (360,240), interpolation=cv2.INTER_AREA)
        gt = cv2.resize(gt, (360,240), interpolation=cv2.INTER_AREA)

        if img.dtype == np.uint8:
            img = (img / 255.0).astype('float32')
        if gt.dtype == np.uint8:
            gt = (gt / 255.0).astype('float32')

        return [img,gt]
