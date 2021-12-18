from conf import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob

def create_df(data_path):
    df = pd.DataFrame()
    img_list = sorted(glob(os.path.join(data_path,'*','*.png')))
    mask_list = sorted(glob(os.path.join(data_path,'*','*.npy')))
    print(len(img_list), len(mask_list))
    assert len(img_list) == len(mask_list) , 'length of image and mask do not equal'
    
    df['img_path'] = img_list
    df['mask_path'] = mask_list
    df['id'] = df.img_path.apply(lambda x : os.path.split(x)[-1].split('.')[0])
    df['type'] = df.img_path.apply(lambda x : x.split('/')[-2])

    return df

def create_test_df(data_path):
    df = pd.DataFrame()
    img_list = sorted(glob(os.path.join(data_path,'*','*.png')))
    
    df['img_path'] = img_list
    df['id'] = df.img_path.apply(lambda x : os.path.split(x)[-1].split('.')[0])
    df['type'] = df.img_path.apply(lambda x : x.split('/')[-2])

    return df

if __name__=="__main__":
    train, valid = create_df(args.tr_path), create_df(args.val_path)
    print("train df: {}".format(train.shape))
    print("valid df: {}".format(valid.shape))
    print(train.head())
    print(valid.head())
    train.to_csv('../data/train.csv', index=False)
    valid.to_csv('../data/valid.csv', index=False)