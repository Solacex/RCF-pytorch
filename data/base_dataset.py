import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
from PIL import Image
import torchvision.transforms.functional as TF
import torch
IMG_MEAN = np.array((122.67891434, 116.66876762, 104.00698793), dtype=np.float32)
class BaseDataSet(data.Dataset):
    def __init__(self, root, list_path, joint_transform=None, transform=None, label_transform = None, max_iters=None, ignore_label=255, set='val', dataset='cityscapes'):
        self.root = root
        self.list_path = list_path
        self.ignore_label = ignore_label
        self.set = set
        self.transform = transform
        self.joint_transform = joint_transform
        self.label_transform = label_transform

        if self.set !='train':
            self.list_path = (self.list_path).replace('train', self.set)

        self.img_ids =[]
        with open(self.list_path) as f:
            for item in f.readlines():
                fields = item.strip().split('\t')[0]
                self.img_ids.append(fields)
        if not max_iters==None:
            total = len(self.img_ids)
            index = list( np.random.choice(total, max_iters, replace=False))
            self.img_ids = [self.img_ids[i] for i in index]
#            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
 #       print(len(self.img_ids))

        self.files = []
        if dataset=='gta5':
            for name in self.img_ids:
                img_file = osp.join(self.root,'images', name)
                label_file = osp.join(self.root,'images', name)
                label_file=label_file.replace('image', 'label')
 #               label_file=label_file.replace('.', '_canny.')
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": name,
                    "centroid":(0,0)
                })
        else:
            for name in self.img_ids:
                img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
                label_name = name.replace('leftImg8bit', 'gtFine_labelIds')#edge)tFine_labelIds.png
                label_file =osp.join(self.root, 'gtFine/%s/%s' % (self.set, label_name))
                self.files.append({
                    "img": img_file,
                    "label":label_file,
                    "name": name,
                    "hard_loc":(0,0)
                })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        try:
 #       if 1:
            image = Image.open(datafiles["img"]).convert('RGB')
            label = Image.open(datafiles["label"]).convert('L')
            name = datafiles["name"]


            centroid=None
            if self.joint_transform is not None:
                image, label = self.joint_transform(image, label, centroid)
            image = image-IMG_MEAN
            image = TF.to_tensor(image)
            if self.transform is not None:
                image = self.transform(image)
            if self.label_transform is not None:
                label = self.label_transform(label)
            wei = torch.ones_like(label).float()
 #       else:
        except Exception as e:
            index = index - 1 if index > 0 else index + 1
            return self.__getitem__(index)
#        if self.set=='train':
         #   return image, label
        #else:
        return image, name


if __name__ == '__main__':
    dst = GTA5DataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
