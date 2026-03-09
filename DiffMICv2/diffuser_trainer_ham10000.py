"""
DiffMICv2 Diffuser Trainer for HAM10000 Dataset.

Adapted from diffuser_trainer.py for HAM10000 with:
    - 7 classes, batch_size 8, 30 training epochs
    - Frozen EfficientSAM-ViT-S backbone
    - BottleneckKANHead-based DCG guidance
    - Validation with F1, AUC, Accuracy, Precision, Recall, Kappa
    - Checkpointing every 10 epochs
    - No progress bar (prints epoch + loss only)
    - No DataParallel — uses single GPU or DDP only
    
References:
    - FastKAN (ZiyaoLi): https://github.com/ZiyaoLi/fast-kan
    - E-BayesSAM (MICCAI 2025): https://link.springer.com/chapter/10.1007/978-3-032-05185-1_13
    - DiffMIC (MICCAI 2023): https://arxiv.org/abs/2303.10610
    - EfficientSAM: https://arxiv.org/abs/2312.00863
    - FastKAN paper: https://arxiv.org/abs/2404.19756
"""

from typing import Optional
import os
# Ensure all relative paths resolve from the script's own directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
import numpy as np
import copy
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

import pytorch_lightning as pl
import yaml
from easydict import EasyDict
import random
from pytorch_lightning import callbacks
from pytorch_lightning.accelerators import accelerator
from pytorch_lightning.core.hooks import CheckpointHooks
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor, EarlyStopping, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import Namespace

from torch.utils.data import DataLoader
import pipeline

from torchvision.utils import save_image
from torchvision.models import vgg16
output_dir = 'logs'
version_name = 'HAM10000_Baseline'
logger = TensorBoardLogger(name='ham10000', save_dir=output_dir)
import math
from pretraining.dcg import DCG as AuxCls
from model import *
from utils import *


class CoolSystem(pl.LightningModule):
    
    def __init__(self, hparams):
        super(CoolSystem, self).__init__()

        self.params = hparams
        self.epochs = self.params.training.n_epochs
        self.initlr = self.params.optim.lr

        
        config_path = r'option/diff_DDIM.yaml'
        with open(config_path, 'r') as f:
            params = yaml.safe_load(f)
        config = EasyDict(params)
        self.diff_opt = config

        self.model = ConditionalModel(self.params, guidance=self.params.diffusion.include_guidance)
        self.aux_model = AuxCls(self.params)
        self.init_weight(ckpt_path='/kaggle/input/datasets/deadlydracula/ham1000-dcg/ham10000_aux_model_epoch20.pth')
        self.aux_model.eval()

        # Freeze aux_model entirely
        for param in self.aux_model.parameters():
            param.requires_grad = False

        self.save_hyperparameters()
        
        self.gts = []
        self.preds = []

        self.DiffSampler = pipeline.SR3Sampler(
            model=self.model,
            scheduler = pipeline.create_SR3scheduler(self.diff_opt['scheduler'], 'train'),
        )
        self.DiffSampler.scheduler.set_timesteps(self.diff_opt['scheduler']['num_test_timesteps'])
        self.DiffSampler.scheduler.diff_chns = self.params.data.num_classes

    def configure_optimizers(self):
        # REQUIRED
        optimizer = get_optimizer(self.params.optim, filter(lambda p: p.requires_grad, self.model.parameters()))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=self.initlr * 0.01)

        return [optimizer], [scheduler]


    def init_weight(self, ckpt_path=None):
        
        if ckpt_path and os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)[0]
            checkpoint_model = checkpoint
            state_dict = self.aux_model.state_dict()
            # 1. filter out unnecessary keys
            checkpoint_model = {k: v for k, v in checkpoint_model.items() if k in state_dict.keys()}
            print(f"Loaded {len(checkpoint_model)} keys from {ckpt_path}")
            # 2. overwrite entries in the existing state dict
            state_dict.update(checkpoint_model)
            
            self.aux_model.load_state_dict(state_dict) 
        else:
            print(f"Warning: checkpoint not found at {ckpt_path}, using random init for aux_model")

    def diffusion_focal_loss(self, prior, targets, noise, noise_gt, gamma=1, alpha=10):
        probs = F.softmax(prior, dim=1)
        probs = (probs * targets).sum(dim=1)
        weights = 1+alpha*(1 - probs) ** gamma
        weights = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        loss = weights*(noise-noise_gt).square()
        return loss.mean()



    def guided_prob_map(self, y0_g, y0_l, bz, nc, np):
    
        distance_to_diag = torch.tensor([[abs(i-j)  for j in range(np)] for i in range(np)]).to(self.device)

        weight_g = 1 - distance_to_diag / (np-1)
        weight_l = distance_to_diag / (np-1)
        interpolated_value = weight_l.unsqueeze(0).unsqueeze(0) * y0_l.unsqueeze(-1).unsqueeze(-1) + weight_g.unsqueeze(0).unsqueeze(0) * y0_g.unsqueeze(-1).unsqueeze(-1)
        diag_indices = torch.arange(np)
        map = interpolated_value.clone()
        for i in range(bz):
            for j in range(nc):
                map[i,j,diag_indices,diag_indices] = y0_g[i,j]
                map[i,j, np-1, 0] = y0_l[i,j]
                map[i,j, 0, np-1] = y0_l[i,j]
        return map

    def training_step(self, batch, batch_idx):
        self.model.train()
        self.aux_model.eval()
        
        x_batch, y_batch = batch
        y_batch, _ = cast_label_to_one_hot_and_prototype(y_batch, self.params)
        y_batch = y_batch.cuda()
        x_batch = x_batch.cuda()
        with torch.no_grad():
            y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = self.aux_model(x_batch)
        
        # attn_map is now (B, num_classes) from KAN head — no spatial dims
        bz, nc = y0_aux_global.size()
        bz, np = attns.size()
        
        y_map = y_batch.unsqueeze(1).expand(-1,np*np,-1).reshape(bz*np*np,nc)
        noise = torch.randn_like(y_map).to(self.device)
        timesteps = torch.randint(0, self.DiffSampler.scheduler.config.num_train_timesteps, (bz*np*np,), device=self.device).long()

        noisy_y = self.DiffSampler.scheduler.add_noise(y_map, timesteps=timesteps, noise=noise)
        noisy_y = noisy_y.view(bz,np*np,-1).permute(0,2,1).reshape(bz,nc,np,np)
        
        y0_cond = self.guided_prob_map(y0_aux_global,y0_aux_local,bz,nc,np)
        y_fusion = torch.cat([y0_cond, noisy_y],dim=1)

        attns = attns.unsqueeze(-1)
        attns = (attns*attns.transpose(1,2)).unsqueeze(1)
        noise_pred = self.model(x_batch, y_fusion, timesteps, patches, attns)

        noise = noise.view(bz,np*np,-1).permute(0,2,1).reshape(bz,nc,np,np)
        loss = self.diffusion_focal_loss(y0_aux,y_batch,noise_pred,noise)

        self.log("train_loss",loss,prog_bar=False)
        return {"loss":loss}

    def on_validation_epoch_end(self):
        gt = torch.cat(self.gts)
        pred = torch.cat(self.preds)
        ACC, BACC, Prec, Rec, F1, AUC_ovo, kappa = compute_isic_metrics(gt, pred)

        self.log('accuracy',ACC)
        self.log('f1',F1)
        self.log('Precision',Prec)        
        self.log('Recall',Rec)
        self.log('AUC',AUC_ovo)
        self.log('kappa',kappa)   
        
        self.gts = []
        self.preds = []
        print(f"Val: Accuracy={ACC:.4f}, F1={F1:.4f}, Precision={Prec:.4f}, "
              f"Recall={Rec:.4f}, AUROC={AUC_ovo:.4f}, Kappa={kappa:.4f}")


    def validation_step(self, batch, batch_idx):
        self.model.eval()
        self.aux_model.eval()

        
        x_batch, y_batch = batch
        y_batch, _ = cast_label_to_one_hot_and_prototype(y_batch, self.params)
        y_batch = y_batch.cuda()
        x_batch = x_batch.cuda()
        with torch.no_grad():
            y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = self.aux_model(x_batch)

        bz, nc = y0_aux_global.size()
        bz, np = attns.size()

        
        y0_cond = self.guided_prob_map(y0_aux_global,y0_aux_local,bz,nc,np)
        yT = self.guided_prob_map(torch.rand_like(y0_aux_global),torch.rand_like(y0_aux_local),bz,nc,np)
        attns = attns.unsqueeze(-1)
        attns = (attns*attns.transpose(1,2)).unsqueeze(1)
        y_pred = self.DiffSampler.sample_high_res(x_batch,yT,conditions=[y0_cond, patches, attns])
        y_pred = y_pred.reshape(bz, nc, np*np)
        y_pred = y_pred.mean(2)
        self.preds.append(y_pred)
        self.gts.append(y_batch)

    
    def train_dataloader(self):
        data_object, train_dataset, test_dataset = get_dataset(self.params)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.params.training.batch_size,
            shuffle=True,
            num_workers=self.params.data.num_workers,
            pin_memory=True,
            drop_last=True
        )
        return train_loader
    
    def val_dataloader(self):
        data_object, train_dataset, test_dataset = get_dataset(self.params)

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.params.testing.batch_size,
            shuffle=False,
            num_workers=self.params.data.num_workers,
            pin_memory=True
        )
        return test_loader  


def main():
    RESUME = False
    resume_checkpoint_path = r'logs/ham10000/version_0/checkpoints/last.ckpt'
    if RESUME == False:
        resume_checkpoint_path = None

    seed = 10
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    config_path = r'configs/ham10000.yml'
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    config = EasyDict(params)

    model = CoolSystem(config)

    checkpoint_callback = ModelCheckpoint(
        monitor='f1',
        filename='ham10000-epoch{epoch:02d}-accuracy-{accuracy:.4f}-f1-{f1:.4f}',
        auto_insert_metric_name=False,   
        every_n_epochs=10,
        save_top_k=1,
        mode="max",
        save_last=True
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        check_val_every_n_epoch=
        num_sanity_val_steps=0,
        max_epochs=config.training.n_epochs,
        accelerator='gpu',
        devices=2,
        precision=32,
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=True),
        enable_progress_bar=False,
        log_every_n_steps=5,
        callbacks=[checkpoint_callback, lr_monitor_callback]
    ) 

    # train
    trainer.fit(model, ckpt_path=resume_checkpoint_path)
    
    
if __name__ == '__main__':
    main()
