import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datetime import datetime
import psutil
import torch
import cv2

abs_path = os.path.dirname(__file__)
n_jobs = psutil.cpu_count()
ctime = datetime.today().strftime("%m%d_%H%M")

args = {
    "Competition":"Heart21",
    "DEBUG":False,
    "SEED":42,
    "n_folds":5,
    "epochs":100, 
    "image_size":[512, 512],
    "num_classes":1,
    "tr_bs":4,
    "val_bs":16,
    "num_workers":n_jobs,
    "arch":'DeepLabV3', # Unet, DeepLabV3, UnetPlusPlus
    "encoder_name":"timm-efficientnet-b0",#"mobilenet_v2", "xception"
    "encoder_weights":'imagenet', # 'imagenet', ssl'(Semi-supervised), 'swsl'(Semi-weakly supervised)
    "loss":"BCELoss", # DiceLoss, DiceBCELoss, IoULoss, FocalLoss, TverskyLoss, BCELoss
    "optimizer":"RAdam", # Adam, RAdam, AdamW, SGD, Lookahead
    "scheduler":"ReduceLROnPlateau", # Cosine, Steplr, Lambda, Plateau, WarmupV2, CosWarm
    "combine_schedulers":["Plateau","Cosine"],
    "patience": 3,
    'es_patience': 20, # earlystop patience
    "lr":1e-3,
    "T_max":9,
    "eta_min":1e-6,
    "plateau_factor":0.9,
    "weight_decay":1e-4,
    "augment_ratio":0.5,
    "amp":False,
}

args['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args['tr_path'] = '../data/train'
args['val_path'] = '../data/validation'
args['tr_df'] = '../data/train.csv'
args['val_df'] = '../data/valid.csv'
args['weight_path'] = './model' # for inference?
args['ckpt_path'] = f'../model/{ctime}_{args["encoder_name"]}_{args["arch"]}'

if args['DEBUG']:
    args['image_size'] = (64,64)
    args['epochs'] = 10

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

args['tr_aug'] = A.Compose([
                    # A.CenterCrop(422, 422, p=1), # 422 is min size
                    A.Resize(*args['image_size'], interpolation=cv2.INTER_LANCZOS4), 
                    A.CLAHE(always_apply=True, p=1),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5),
                    # A.VerticalFlip(p=0.5),
                    # A.RandomRotate90(p=0.5),
                    ToTensorV2(),
                    ])

args['val_aug'] = A.Compose([
                    A.Resize(*args['image_size']),
                    A.CLAHE(always_apply=True, p=1),
                    ToTensorV2(),
                  ])

args['test_aug'] = A.Compose([
                    A.Resize(*args['image_size'], interpolation=cv2.INTER_LANCZOS4), 
                    # A.HorizontalFlip(),
                    # A.GridDistortion(p=0.2)
                  ])
