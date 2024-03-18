import json 
import os.path
import os
from itertools import chain
import random 
import torch
from torch import optim
from einops import rearrange
import pandas as pd
import numpy as np
from lr_scheduler import LR_Scheduler
from meanshift import MeanShiftCluster
from tqdm import tqdm 
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

class Trainer():
    def __init__(self, 
                 fold, 
                 writer,
                 train_loader,
                 val_loader,
                 model,
                 criterion,
                 metric,
                 args
                 ):
        self.args = args
        self.model = model
        self.metric = metric
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.writer = writer
        self.fold = fold
        self.dict_record = {'epoch' : {}}
        
        self.meanshift = MeanShiftCluster()
        self.init_optim()
        self.training()


    def init_optim(self):
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, 
                   self.model.parameters()), 
                   lr=self.args.lr, weight_decay=1e-5)
        self.scheduler = LR_Scheduler(self.args.lr_scheduler, self.args.lr,
                                      self.args.epochs, len(self.train_loader), 
                                      min_lr=self.args.min_lr)
        
    def training(self):

        for epoch in range(self.args.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.val_epoch(epoch)
            metric_summaruy = self.metric.summary()
            train_metric =  metric_summaruy["Train"]["Mean"]
            val_metric = metric_summaruy["Evaluation"]["Mean"]

            self.writer.add_scalar(f'training_loss_fold-{self.args.fold}', train_loss, epoch)
            # self.writer.add_scalar('training_dice_fold'+str(self.fold), train_metric, epoch)
            self.writer.add_scalar(f'lr_fold-{self.args.fold}',
                                   self.optimizer.param_groups[0]['lr'], epoch)
            # self.writer.add_scalar('validate_d_fold'+str(self.fold), val_metric, epoch)
            self.writer.add_scalar(f'val_loss_fold-{self.args.fold}', val_loss, epoch)
            self.writer.add_scalar(f'train_metric_fold-{self.args.fold}',
                                   train_metric, epoch)
            self.writer.add_scalar(f'val_metric_fold-{self.args.fold}',
                                    val_metric, epoch)
            
            print('Epoch: {0}\t' 
                  'Training Loss {train_loss:.4f} \t'
                  'Val Loss {val_loss:.4f} \t'
                  'Training Metric  {train_metric:.4f}\t'
                  'Val Metric  {val_metric:.4f}\t'
                  .format(epoch, train_loss=train_loss, 
                                 val_loss=val_loss,
                                 train_metric=train_metric,
                                 val_metric=val_metric
                                 ))
            
            if epoch % self.args.num_epoch_record == 0:
                torch.save({
                    'epoch' : epoch,
                    'model_state_dic' : self.model.state_dict(),
                    "optimizer_state_dict" : self.optimizer.state_dict(),
                    },
                    os.path.join(self.args.checkpoints_dir, "epoch-{}.pth".format(epoch)))

    def val_epoch(self, epoch):
        self.model.eval()
        self.metric.reset_val()
        with torch.no_grad():
            for batch_idx, tup in enumerate(self.val_loader):
                img, label, keypoints = tup
                image_var = img.float().to(self.args.device)
                label = label.float().to(self.args.device).unsqueeze(1)
                keypoints = keypoints.float().to(self.args.device)
                # label_logits, _ = self.model(image_var, keypoints)
                label_logits = self.model(image_var)
                loss = self.criterion(label_logits, label)
                seg = self.meanshift(label_logits)
                seg = rearrange(seg, 'b h w -> b (h w)')
                #seg = (seg - seg.min(dim=-1)) / (seg.max(dim=-1) - seg.min(dim=-1)) * 255
                label = rearrange(label.squeeze(1), 'b h w -> b (h w)')
                self.metric.add_val(seg.detach().cpu(), label.detach().cpu())

        self.visualize_seg(epoch=epoch, type="val")    
        return loss 


    def visualize_seg(self, epoch=0, type="train"):
        
        if type == "train":
            data_loader = self.train_loader
        elif type == "val":
            data_loader = self.val_loader

        with torch.no_grad():
            img, label, keypoints = next(iter(data_loader))  # Example: get first batch
            img = img.float().to(self.args.device)
            label = label.float().to(self.args.device)
            keypoints = keypoints.float().to(self.args.device)
            # label_logits, _ = self.model(img, keypoints)
            label_logits = self.model(img)
            
            seg = self.meanshift(label_logits).cpu().numpy()[0]
            img = img[0].squeeze(0).cpu().numpy()
            label = label[0].cpu().numpy()

            img_trans = plt.get_cmap('gray')(img/img.max())[:, :, :3]  # Discard alpha
            label_trans = plt.get_cmap('tab20')(label/label.max())[:, :, :3]  # Discard alpha
            seg_trans = plt.get_cmap('tab20')(seg/seg.max())[:, :, :3]  # Discard alpha

            # Create an overlay where the label is not zero
            label_trans = np.where(label.reshape((512, 512, 1)) != 0, label_trans, img_trans)
            seg_trans = np.where(seg.reshape((512, 512, 1)) != 0, seg_trans, img_trans)
            
            # (H, W, C) --> (C, H, W)
            img_trans = torch.tensor(img_trans.transpose((2, 0, 1))).float()
            label_trans = torch.tensor(label_trans.transpose((2, 0, 1))).float()
            seg_trans = torch.tensor(seg_trans.transpose((2, 0, 1))).float()
            
            self.writer.add_image(
                os.path.join(self.args.save_img_dir, f"epoch-{epoch}_{type}.png"),
                  make_grid([img_trans, label_trans, seg_trans]),
                  epoch)
            
            fig, axes = plt.subplots(1, 3)
            axes[0].imshow(img, cmap="gray")
            axes[1].imshow(img, cmap="gray")
            axes[2].imshow(img, cmap="gray")
            seg_masked = np.ma.masked_where(seg.reshape((512,512)) == 0, (seg.reshape((512,512))))
            label_masked = np.ma.masked_where(label.reshape((512,512)) == 0, (label.reshape((512,512))))
            axes[1].imshow(label_masked, cmap="tab20")
            axes[2].imshow(seg_masked, cmap="tab20")
            plt.axis("off")
            plt.savefig(os.path.join(self.args.save_img_dir, f"epoch-{epoch}_{type}.png"))  



    def train_epoch(self, epoch):
        self.model.train()
        self.metric.reset_train()

        for batch_idx, tup in tqdm(enumerate(self.train_loader)):
            img, label, keypoints = tup
            image_var = img.float().to(self.args.device)
            label = label.float().to(self.args.device).unsqueeze(1)
            keypoints = keypoints.float().to(self.args.device)
            self.scheduler(self.optimizer, batch_idx, epoch)
            # label_logits, _ = self.model(image_var, keypoints)
            label_logits = self.model(image_var)
            loss = self.criterion(label_logits, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                seg = self.meanshift(label_logits)
                seg = rearrange(seg, 'b h w -> b (h w)')
                #seg = (seg - seg.min(dim=-1)) / (seg.max(dim=-1) - seg.min(dim=-1)) * 255
                label = rearrange(label.squeeze(1), 'b h w -> b (h w)')
                self.metric.add_train(seg.detach().cpu(), label.detach().cpu())

        self.visualize_seg(type="train")    
        return loss




        
