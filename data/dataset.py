from .gta5_dataset import GTA5DataSet
from .synthia_dataset import SYNTHIADataSet
from .cityscapes_dataset import cityscapesDataSet
import numpy as np
from torch.utils.data import DataLoader
from . import joint_transforms
from .base_dataset import BaseDataSet
import torchvision.transforms as standard_transforms
from . import transforms
from torch.utils import data
import torchvision.transforms.functional as TF

IMG_MEAN = np.array((122.67891434, 116.66876762, 104.00698793), dtype=np.float32)
def get_dataset(args, input_size_source, input_size_target):
    targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                     max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                     crop_size=input_size_target,
                                                     scale=False, mirror=False, mean=IMG_MEAN,
                                                     set=args.set),
                                   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                   pin_memory=True)
    targetloader_iter = enumerate(targetloader)
    return targetloader_iter
def get_hed_dataset(b, dataset):
    joint_list = [
            joint_transforms.RandomSizeAndCrop(500,
                                           True,
                                           scale_min=0.5,
                                           scale_max=1.5,
                                           pre_size=720
                                           ),
            joint_transforms.Resize(500),
            joint_transforms.RandomHorizontallyFlip()
            ]
    joint_transform = joint_transforms.Compose(joint_list)
    mean = [float(item) / 255.0 for item in IMG_MEAN]
    std  = [1,1,1]
    train_transform_list = [
            standard_transforms.Normalize(mean, std)
            ]
    train_transform = None#standard_transforms.Compose(train_transform_list)
    label_transform = transforms.MaskToTensor()
    if dataset=='cityscapes':
        dataset = BaseDataSet('/home/guangrui/data/cityscapes', './data/cityscapes_list/train.txt', 
                        joint_transform =  joint_transform,
                        transform = train_transform,
                        label_transform = label_transform,
                        set='train', dataset=dataset)
    elif dataset=='gta5':
        dataset = BaseDataSet('/home/guangrui/data/gta5', './data/gta5_list/crst_train.txt',
                        joint_transform =  joint_transform,
                        transform = train_transform,
                        label_transform = label_transform,
                        set='train', dataset=dataset)

    loader = data.DataLoader(dataset, batch_size=b, shuffle=True, num_workers=4, pin_memory=True)
    return loader
def training_dataset():
    joint_list = [
            joint_transforms.Resize((1440, 720))
            ]
    joint_transform = joint_transforms.Compose(joint_list)

    mean = [float(item) / 255.0 for item in IMG_MEAN]
    std  = [1,1,1]
    train_transform_list = [
            transforms.ToTensor(),
            standard_transforms.Normalize(mean, std)
            ]

    train_transform = None#standard_transforms.Compose(train_transform_list)
    label_transform = transforms.MaskToTensor()
    dataset = BaseDataSet('/home/guangrui/data/gta5', './data/gta5_list/train.txt',
                        joint_transform =  joint_transform,
                        transform = train_transform,
                        label_transform = label_transform,
                        set='train', dataset='gta5')
    loader = data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    return loader

def get_hed_test_dataset(b):
    joint_list = [
            joint_transforms.Resize((1440, 720))
            ]
    joint_transform = joint_transforms.Compose(joint_list)

    mean = [float(item) / 255.0 for item in IMG_MEAN]
    std  = [1,1,1]
    train_transform_list = [
            transforms.ToTensor(),
            standard_transforms.Normalize(mean, std)
            ]

    train_transform = None#standard_transforms.Compose(train_transform_list)
    label_transform = transforms.MaskToTensor()
    dataset = BaseDataSet('/home/guangrui/data/cityscapes', './data/cityscapes_list/train.txt', 
                        joint_transform =  joint_transform,
                        transform = train_transform,
                        label_transform = label_transform,
                        set='train')
    loader = data.DataLoader(dataset, batch_size=b, shuffle=True, num_workers=4, pin_memory=True)
    return loader

def init_dataset(cfg, env, plabel_path=None, selected=None):
    source_env = env[cfg.source]
    target_env = env[cfg.target]
    cfg.num_classes=19
    cfg.source_size = source_env.input_size
    cfg.target_size = target_env.input_size
    cfg.source_data_dir  = source_env.data_dir
    cfg.source_data_list = source_env.data_list
    cfg.target_data_dir  = target_env.data_dir
    cfg.target_data_list = target_env.data_list
    source_joint_list = [       
            joint_transforms.RandomSizeAndCrop(cfg.crop_src,
                                                True,
                                                scale_min=cfg.scale_min,
                                                scale_max=cfg.scale_max,
                                                pre_size=cfg.input_src
                                                ),
            joint_transforms.Resize(cfg.crop_src)
            ]

    target_joint_list = [
            joint_transforms.RandomSizeAndCrop(cfg.crop_tgt,
                                           True,
                                           scale_min=cfg.scale_min,
                                           scale_max=cfg.scale_max,
                                           pre_size=cfg.input_tgt
                                           ),
            joint_transforms.Resize(cfg.crop_tgt)
            ]

    if cfg.mirror:
        source_joint_list.append(joint_transforms.RandomHorizontallyFlip())
        target_joint_list.append(joint_transforms.RandomHorizontallyFlip())


    target_joint_transform = joint_transforms.Compose(target_joint_list)
    source_joint_transform = joint_transforms.Compose(source_joint_list)

    train_transform_list = [
            transforms.FlipChannels(),
            transforms.SubMean(),
            transforms.ToTensor()
            ]

    train_transform = standard_transforms.Compose(train_transform_list)
    label_transform = transforms.MaskToTensor()
    trainloader = data.DataLoader(
            BaseDataSet(source_env.data_dir, source_env.data_list, max_iters=cfg.num_steps * cfg.batch_size,
                        joint_transform =  source_joint_transform,
                        transform = train_transform,
                        label_transform = label_transform,
                        set='train',dataset='gta5'),
            batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.worker, pin_memory=True)
    targetloader = data.DataLoader(
            BaseDataSet(target_env.data_dir, target_env.data_list, max_iters=cfg.num_steps * cfg.batch_size,
                        joint_transform =  target_joint_transform,
                        transform = train_transform,
                        label_transform = label_transform,
                        set='train',dataset='cityscapes', plabel_path=plabel_path),
            batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.worker, pin_memory=True)
    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)


    return trainloader_iter, targetloader_iter, cfg


def init_test_dataset(args, dataset_name, set, selected=None, prop=None, label_ori=None):
    max_prop = None
    if dataset_name=='gta5' and set=='train':
        max_prop = args.pool_prop
    if prop is not None:
        max_prop = prop
    if selected is not None:
        max_prop=None

    env = args[dataset_name]
    
    if label_ori is None:
        if dataset_name == 'gta5':
            label_ori=False
        else:
            label_ori=True

    if not label_ori:
        joint_transform = [joint_transforms.Resize((1024, 512))]
        joint_transform = joint_transforms.Compose(joint_transform)
        transform_list = [
            transforms.FlipChannels(),
            transforms.SubMean(),
            transforms.ToTensor()
            ]
    else:
        joint_transform = None
        transform_list = [
            transforms.Resize((1024, 512)),
            transforms.FlipChannels(),
            transforms.SubMean(),
            transforms.ToTensor()
            ]

    train_transform = standard_transforms.Compose(transform_list)
    label_transform = transforms.MaskToTensor()

    targetloader = data.DataLoader(
            BaseDataSet(env.data_dir, env.data_list, 
                        joint_transform =  joint_transform,
                        transform = train_transform,
                        label_transform = label_transform,
                        max_prop=max_prop,
                        selected=selected,
                        set=set, dataset=dataset_name),
            batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    return targetloader
