from conf import *
from inferece import inference, metric_dice_ji
from unet_model import SegModel
from dataset import HearTTestDataset
from trans import get_transforms
from post_pro import crf
import os
import cv2
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Optional
from create_df import create_test_df
import matplotlib.pyplot as plt

class LitClassifier(pl.LightningModule):
    """
    >>> LitClassifier(Backbone())  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    LitClassifier(
      (backbone): ...
    )
    """

    def __init__(
        self,
        scale_list = [0.25, 0.5], # 0.125, 
        backbone: Optional[nn.Module] = None,
        learning_rate: float = 0.0001,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['backbone'])
        if backbone is None:
            backbone = SegModel()
        self.backbone = backbone        

    def forward(self, batch):
        output = self.backbone.model(batch) # 
        # HR OCR network 
        # output = F.interpolate(input=output[0], size=(512, 512), mode='bilinear', align_corners=True)
        return output

    

if __name__ == '__main__':
    # model = LitClassifier(backbone=SegModel(encoder_name='tu-hrnet_w18'))
    # print(model.backbone)

    # Load Data
    if args.DEBUG:
        test_df = create_test_df(args.test_path)[:64]
        # valid = pd.read_csv('../data/valid.csv')[:64]
    else:
        test_df = create_test_df(args.test_path)
    print(test_df.head())
    test_dataset = HearTTestDataset(test_df, 
                                    base_path=args.test_path, 
                                    transform=get_transforms(args, data='valid'))
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # read mask
    # mask_list = valid_dataset.masks

    # read img 
    # img_list = test_dataset.imgs
    # mask_list = list()
    # for i in tqdm(range(len(valid))):
    #     mask = valid_dataset.read_mask(valid, i, valid_dataset.base_path)
    #     mask_list.append(mask)

    pred_list = list()

    for path in args.weights:
    
        # path = '../model/resnet18_UnetPlusPlus_9575/epoch=77-val_dice_score=0.9575_.ckpt'
        encoder_name = path.split("/")[2].split("_")[0]
        arch_name = path.split("/")[2].split("_")[1]
        print("Inference ... ",encoder_name, arch_name)

        model = LitClassifier()
        model = model.load_from_checkpoint(path, backbone=SegModel(encoder_name=encoder_name, arch_name=arch_name)).cuda().eval()

        pred = inference(test_dataloader, model)
        pred_list.append(pred)


    pred = np.array(pred_list)
    pred = pred.mean(axis=0)

    test_path = test_dataset.df.img_path.values
    img_list = test_dataset.imgs

    # rescale to original and save
    for src_path, img, out in zip(test_path, img_list, pred):
        out = cv2.resize(out[0], img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        out = crf(img, out)
        
        new_file_name = src_path.replace('test','submission') # ../data/submission/A2C/0901.png
        new_dir = os.path.split(new_file_name)[0] # ../data/submission/A2C/
        os.makedirs(new_dir, exist_ok=True)
        
        cv2.imwrite(new_file_name,out)
        # plt.imsave(new_file_name,out)

    # metric_dice_ji(pred, mask_list, img_list, 0.5)