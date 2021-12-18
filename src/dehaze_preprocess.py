from glob import glob
import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

import cv2;
import math;
import numpy as np;

def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz);
    imvec = im.reshape(imsz,3);

    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.95;
    im3 = np.empty(im.shape,im.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz);
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
    cov_Ip = mean_Ip - mean_I*mean_p;

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
    var_I   = mean_II - mean_I*mean_I;

    a = cov_Ip/(var_I + eps);
    b = mean_p - a*mean_I;

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));

    q = mean_a*im + mean_b;
    return q;

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray)/255;
    r = 60;
    eps = 0.0001;
    t = Guidedfilter(gray,et,r,eps);

    return t;

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype);
    t = cv2.max(t,tx);

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res

def make_dehaze(img_path, mode):
    size = 50
    src = cv2.imread(img_path)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    I = src.astype('float64')/255;
    
    dark = DarkChannel(I,size);
    A = AtmLight(I,dark);
    te = TransmissionEstimate(I,A,15);
    t = TransmissionRefine(src,te);
    J = Recover(I,t,A,0.1);
    new_path = img_path.replace(f'{mode}',f'{mode}_dehaze')
    os.makedirs(os.path.split(new_path)[0], exist_ok=True)
    J = J*255
#     J = J.astype('float64')
    cv2.imwrite(new_path, J)
#     plt.imshow(J)
#     plt.savefig(new_path)
#     return J


def create_col(path, df, mode):
    img_list = glob(path + '/*/*png')
    if mode == 'train':
        mask_path = path.replace('train_dehaze','train')
    else:
        mask_path = path.replace('validation_dehaze','validation')
    mask_list = glob(mask_path + '/*/*npy')
    
    df['img_path'] = sorted(img_list)
    df['mask_path'] = sorted(mask_list)
    if mode == 'train':
        df['mask_path'] = df['mask_path'].apply(lambda x: x.replace('train','train_dehaze'))
    else:
        df['mask_path'] = df['mask_path'].apply(lambda x: x.replace('validation','validation_dehaze'))
    df['id'] = df.img_path.apply(lambda x : x.split('/')[-1].split('.')[0])
    df['type'] = df.img_path.apply(lambda x : x.split('/')[-2])

    return df

def create_df():
    train_path = '../data/train_dehaze/'
    train = pd.DataFrame()
    train = create_col(train_path, train, 'train')

    valid_path = '../data/validation_dehaze/'
    valid = pd.DataFrame()
    valid = create_col(valid_path, valid,'validation')
    return train, valid


mode = ['train','validation']
data_type = ['A2C','A4C']
for m in mode:
    data_path = f'../data/{m}'
    img_list = glob(os.path.join(data_path,'*','*.png'))
    for p in tqdm(img_list):
        make_dehaze(p, m) # train, validation

# copy label to dehaze dir

for m in mode:
    for dtype in data_type:
        data_path = f'../data/{m}/{dtype}/'
        new_path = f'../data/{m}_dehaze/{dtype}/'
        label_list = glob(os.path.join(data_path,'*.npy'))
        for l in tqdm(label_list):
            shutil.copy2(l, new_path)

train, valid = create_df()
print(train.head())
print(valid.head())

train.to_csv('../data/train_dehaze.csv', index=False)
valid.to_csv('../data/valid_dehaze.csv', index=False)

