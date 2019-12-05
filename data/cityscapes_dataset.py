import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
from PIL import Image
import random


class cityscapesDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=False, mirror=True, ignore_label=255, set='val'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        #self.mean_bgr = np.array([72.30608881, 82.09696889, 71.60167789])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        self.id2train = { 7:0, 8: 1, 11: 2, 12:3,  13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10,
                24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}     # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            label_name = name.replace('leftImg8bit', 'gtFine_labelIds')
            label_file =osp.join(self.root, 'gtFine/%s/%s' % (self.set, label_name))
            self.files.append({
                "img": img_file,
                "label":label_file,
                "name": name
            })
    
    def __scale__(self):
        cropsize = self.crop_size
        if self.scale:
            r = random.random()
            if r > 0.7:
                cropsize = (int(self.crop_size[0] * 1.1), int(self.crop_size[1] * 1.1))
            elif r < 0.3:
                cropsize = (int(self.crop_size[0] * 0.8), int(self.crop_size[1] * 0.8))

        return cropsize
            
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        cropsize = self.__scale__()

        try:
            image = Image.open(datafiles["img"]).convert('RGB')
            label = Image.open(datafiles["label"])
            name = datafiles["name"]
                
            # resize
            image = image.resize(cropsize, Image.BICUBIC)
            if self.set!='val':
                label = label.resize(self.crop_size, Image.NEAREST)
            image = np.asarray(image, np.float32)
            label = np.asarray(label, np.float32)

            label_copy = 255 * np.ones(label.shape, dtype=np.int32)
            for k, v in self.id2train.items():
                label_copy[label == k] = v
            label = np.asarray(label_copy, dtype=np.float32)
            size = image.shape
            size_l = label.shape
            image = image[:, :, ::-1]  # change to BGR
            image -= self.mean
            image = image.transpose((2, 0, 1)).copy()
    
    
            if self.is_mirror and random.random() < 0.5:
                idx = [i for i in range(size[1] - 1, -1, -1)]
                idx_l = [i for i in range(size_l[1] - 1, -1, -1)]
                image = np.take(image, idx, axis = 2)
                label = np.take(label, idx_l, axis=1)

        except Exception as e:
            print(datafiles)
            index = index - 1 if index > 0 else index + 1 
            return self.__getitem__(index)
        
        return image.copy(), label.copy(), np.array(size), np.array(size), name


if __name__ == '__main__':
    dst = GTA5DataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=200)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
