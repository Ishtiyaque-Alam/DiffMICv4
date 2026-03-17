from typing import Optional
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
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
from pytorch_lightning.callbacks import ModelCheckpoint,DeviceStatsMonitor,EarlyStopping,LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import Namespace

from torch.utils.data import DataLoader

from torchvision.utils import save_image
from torchvision.models import vgg16
output_dir = 'logs'
version_name='Baseline'
logger = TensorBoardLogger(name='placental',save_dir = output_dir )
import matplotlib.pyplot as plt
# import tent
import math
from pretraining.dcg import DCG as AuxCls
from model import *
from utils import *
from rectified_flow import RectifiedFlow
from rectified_flow.flow_components.interpolation_solver import AffineInterp


class CoolSystem(pl.LightningModule):
    
    def __init__(self, hparams):
        super(CoolSystem, self).__init__()

        self.params = hparams
        self.epochs = self.params.training.n_epochs
        self.initlr = self.params.optim.lr


        feature_dim = self.params.model.feature_dim
        num_classes = self.params.data.num_classes

        self.feature_encoder = SamEncoder(arch=self.params.model.arch, feature_dim=feature_dim, config=self.params)
        self.condition_proj = nn.Linear(num_classes * 2, feature_dim)
        self.velocity_net = nn.Sequential(
            nn.Linear(feature_dim + 1, feature_dim * 2),
            nn.GELU(),
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        orig_forward = self.velocity_net.forward
        def conditioned_forward(x, t=None):
            if t is None:
                return orig_forward(x)
            if t.dim() == 0:
                t = t.expand(x.size(0))
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            return orig_forward(torch.cat([x, t], dim=-1))
        self.velocity_net.forward = conditioned_forward
        self.flow = RectifiedFlow(
            data_shape=(feature_dim,),
            velocity_field=self.velocity_net,
            interp=AffineInterp('straight')
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

        self.aux_model = AuxCls(self.params)
        self.init_weight(ckpt_path='pretraining/ckpt/placental_aux_model.pth')
        self.aux_model.eval()
        for param in self.aux_model.parameters():
            param.requires_grad = False

        self.save_hyperparameters()
        
        self.gts = []
        self.preds = []

    def configure_optimizers(self):
        # REQUIRED
        params = list(self.feature_encoder.parameters()) + \
                 list(self.condition_proj.parameters()) + \
                 list(self.velocity_net.parameters()) + \
                 list(self.classifier.parameters())
        optimizer = get_optimizer(self.params.optim, filter(lambda p: p.requires_grad, params))
        # optimizer = Lion(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.initlr,betas=[0.9,0.99],weight_decay=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=self.initlr * 0.01)

        return [optimizer], [scheduler]


    def init_weight(self,ckpt_path=None):
        
        if ckpt_path:
            checkpoint = torch.load(ckpt_path,map_location=self.device)[0]
            checkpoint_model = checkpoint
            state_dict = self.aux_model.state_dict()
            # # 1. filter out unnecessary keys
            checkpoint_model = {k: v for k, v in checkpoint_model.items() if k in state_dict.keys()}
            print(checkpoint_model.keys())
            # 2. overwrite entries in the existing state dict
            state_dict.update(checkpoint_model)
            
            self.aux_model.load_state_dict(state_dict) 

    def _sample_flow(self, batch_size, device, num_steps):
        x_t = self.flow.sample_source_distribution(batch_size).to(device)
        t = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        for i in range(num_steps):
            t_i = t[i].expand(batch_size)
            dt = t[i + 1] - t[i]
            v_t = self.flow.get_velocity(x_t, t_i)
            x_t = x_t + dt * v_t
        return x_t

    def training_step(self, batch, batch_idx):
        self.feature_encoder.train()
        self.aux_model.eval()
        
        x_batch, y_batch = batch
        labels = y_batch.to(self.device).long()
        x_batch = x_batch.to(self.device)
        with torch.no_grad():
            y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = self.aux_model(x_batch)

        features = self.feature_encoder(x_batch)
        condition = self.condition_proj(torch.cat([y0_aux_global, y0_aux_local], dim=1))
        x0 = features + condition

        flow_loss = self.flow.get_loss(x_1=x0)
        refined = self._sample_flow(
            batch_size=x0.size(0),
            device=x0.device,
            num_steps=self.params.flow_matching.num_steps
        )
        logits = self.classifier(refined)
        cls_loss = F.cross_entropy(logits, labels)
        total_loss = flow_loss + cls_loss

        self.log("train_loss", total_loss, prog_bar=True)
        return {"loss": total_loss}

    # def validation_step_end(self,step_output):
    #     model_state_dict = self.model.state_dict()
    #     torch.save(model_state_dict, os.path.join(self.save_path,'ckp.pth'))
    #     print('checkpoint save!')
    #     ema_model_state_dict = self.ema_model.state_dict()
    #     for key in model_state_dict:
    #         ema_model_state_dict[key] = 0.999*ema_model_state_dict[key] + 0.001*model_state_dict[key]
    #     self.ema_model.load_state_dict(ema_model_state_dict)
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
        print("Val: Accuracy {0}, F1 score {1}, Precision {2}, Recall {3}, AUROC {4}, Cohen Kappa {5}".format(ACC,F1,Prec,Rec,AUC_ovo,kappa))


    def validation_step(self,batch,batch_idx):
        self.feature_encoder.eval()

        
        x_batch, y_batch = batch
        y_batch, _ = cast_label_to_one_hot_and_prototype(y_batch, self.params)
        y_batch = y_batch.to(self.device)
        x_batch = x_batch.to(self.device)
        refined = self._sample_flow(
            batch_size=x_batch.size(0),
            device=x_batch.device,
            num_steps=self.params.flow_matching.num_steps
        )
        logits = self.classifier(refined)
        y_pred = F.softmax(logits, dim=1)
        self.preds.append(y_pred)
        self.gts.append(y_batch)

        
        # self.log('accuracy',ACC)
        # self.log('f1',F1)
        
        # return {"gt":y_batch,"pred":y_pred}
    
    def train_dataloader(self):
        data_object, train_dataset, test_dataset = get_dataset(self.params)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.params.training.batch_size,
            shuffle=True,
            num_workers=self.params.data.num_workers,
            #sampler=sampler
        )
        return train_loader
    
    def val_dataloader(self):
        data_object, train_dataset, test_dataset = get_dataset(self.params)

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.params.testing.batch_size,
            shuffle=False,
            num_workers=self.params.data.num_workers,
        )
        return test_loader  


def main():
    RESUME = False
    resume_checkpoint_path = r'logs/placental/version_0/checkpoints/last.ckpt'
    if RESUME == False:
        resume_checkpoint_path =None

    seed = 10
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    config_path = r'configs/placental.yml'
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    config = EasyDict(params)


    # hparams = Namespace(**args)

    model = CoolSystem(config)

    checkpoint_callback = ModelCheckpoint(
        monitor='f1',
        filename='placental-epoch{epoch:02d}-accuracy-{accuracy:.4f}-f1-{f1:.4f}',
        auto_insert_metric_name=False,   
        every_n_epochs=1,
        save_top_k=1,
        mode = "max",
        save_last=True
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    precision = "16-mixed" if config.training.mixed_precision else 32
    trainer = pl.Trainer(
        check_val_every_n_epoch=5,
        max_epochs=config.training.n_epochs,
        accelerator='gpu',
        devices=1,
        precision=precision,
        logger=logger,
        strategy="auto",
        enable_progress_bar=True,
        log_every_n_steps=5,
        callbacks = [checkpoint_callback,lr_monitor_callback]
    ) 

    #train
    trainer.fit(model,ckpt_path=resume_checkpoint_path)
    
    #validate
    # val_path=r'DiffMIC/logs/placental/version_4/checkpoints/placental-epoch924-accuracy-0.9350-f1-0.9327.ckpt'
    # trainer.validate(model,ckpt_path=val_path)
    
if __name__ == '__main__':
	#your code
    main()
