import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
import wandb  # 添加wandb导入

torch.set_float32_matmul_precision('medium')
from torch.utils.data import DataLoader
from models import build_model
from datasets import build_dataset
from utils.utils import set_seed, find_latest_checkpoint
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar  # Import ModelCheckpoint and TQDMProgressBar
import hydra
from omegaconf import OmegaConf
import os,sys
 

@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    model = build_model(cfg)

    train_set = build_dataset(cfg)
    val_set = build_dataset(cfg, val=True)

    train_batch_size = max(cfg.method['train_batch_size'] // len(cfg.devices),  1)
    eval_batch_size = max(cfg.method['eval_batch_size'] // len(cfg.devices), 1)

    # 配置WandbLogger，添加更多配置选项
    wandb_logger = None if cfg.debug else WandbLogger(
        project="unitraj",
        name=cfg.exp_name,
        id=cfg.exp_name,
        log_model=True,  # 记录模型检查点
        save_dir=f'/data1/data_zzs/unitraj_ckpt/{cfg.exp_name}',  # 保存日志的目录
        config=cfg,  # 记录配置参数
    )

    # 创建回调函数列表
    call_backs = []

    # 修改checkpoint回调，添加更多指标监控
    checkpoint_callback = ModelCheckpoint(
        monitor='val/brier_fde',
        filename='{epoch}-{val/brier_fde:.2f}-{val/ade:.2f}-{val/fde:.2f}',  # 添加更多指标
        save_top_k=3,  # 保存最好的3个模型
        mode='min',
        dirpath=f'/data1/data_zzs/unitraj_ckpt/{cfg.exp_name}'
    )

    call_backs.append(checkpoint_callback)

    # 添加进度条回调以显示训练进度
    progress_bar = TQDMProgressBar(refresh_rate=1)
    call_backs.append(progress_bar)

    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, num_workers=cfg.load_num_workers, drop_last=False,
        collate_fn=train_set.collate_fn)

    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
        collate_fn=train_set.collate_fn)

    # 配置Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.method.max_epochs,
        logger=wandb_logger,  # 使用配置好的wandb_logger
        devices=1 if cfg.debug else cfg.devices,
        gradient_clip_val=cfg.method.grad_clip_norm,
        accelerator="gpu" if cfg.debug else "gpu",
        profiler="simple",
        strategy="auto" if cfg.debug else "ddp",
        callbacks=call_backs,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=10,  # 每10步记录一次日志
    )

    # automatically resume training
    if cfg.ckpt_path is None and not cfg.debug:
        # Pattern to match all .ckpt files in the base_path recursively
        search_pattern = os.path.join('/data1/data_zzs/unitraj_ckpt', cfg.exp_name, '**', '*.ckpt')
        cfg.ckpt_path = find_latest_checkpoint(search_pattern)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=cfg.ckpt_path)


if __name__ == '__main__':
   # print(sys.path)
    train()
