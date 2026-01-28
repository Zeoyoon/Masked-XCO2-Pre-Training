import os
import argparse
import random
import torch
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from apex import amp

from dataset import Loader_CO2M, raw_data_statistics
from model import MXPT_Transformer
from loss import masked_loss, PlumeRegLoss
from utils import AverageMeter, load_pt_model


def validate_plume_cls(data_loader, model, y_min, y_max):
    """
    Validate the Plume Classification task.
    """
    model.eval()
    scores = []
    with torch.no_grad():
        for i, sample in enumerate(tqdm(data_loader, desc="Validating PlumeCLS")):
            # Complete XCO2 observation sequence
            x = sample["gt_rec"].cuda(non_blocking=True)
            # FFCO2-induced XCO2 concentration
            true_conc = sample["gt_y"].cuda(non_blocking=True)
            
            y_norm = torch.where(
                true_conc <= y_min, 
                torch.zeros_like(true_conc),
                torch.where(
                    true_conc >= y_max, 
                    torch.ones_like(true_conc),
                    (true_conc - y_min) / (y_max - y_min)
                )
            )
            
            _, pred = model(x, None)

            pred = pred.cpu().numpy().squeeze(-1) 
            y_norm = y_norm.cpu().numpy()

            score = mean_absolute_error(y_norm, pred)
            scores.append(score)
            
    return np.array(scores).mean()


def validate_recon(data_loader, model, std, mean):
    """
    Validate the Reconstruction task.
    """
    model.eval()
    maes = []
    with torch.no_grad():
        for i, sample in enumerate(tqdm(data_loader, desc="Validating Reconstruction")):
            x = sample["xco2_data"].cuda(non_blocking=True)
            gt_rec = sample["gt_rec"].cuda(non_blocking=True)
            masked = sample["masked"].cuda(non_blocking=True)
            
            out, _ = model(x, masked)
            
            std_gpu = std.cuda(non_blocking=True)
            mean_gpu = mean.cuda(non_blocking=True)

            out = (out * std_gpu + mean_gpu)
            gt_rec = (gt_rec * std_gpu + mean_gpu)

            # Masking was applied to all channels; compute MAE metric for XCO2 channel only
            masked_xco2 = (masked == 1.0)[:, :, 0]
            gt_rec_xco2 = gt_rec[:, :, 0][masked_xco2].cpu().numpy()
            out_xco2 = out[:, :, 0][masked_xco2].cpu().numpy()
            
            mae = mean_absolute_error(gt_rec_xco2, out_xco2)
            maes.append(mae)

    return np.array(maes).mean()


def get_parser():
    parser = argparse.ArgumentParser(description="Training script for MXPT Transformer")
    parser.add_argument("--mode", choices=['Pre-train', 'PlumeCLS'], default='Pre-train', help="Training mode")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--pt", "--if_load_pt", type=int, default=0, help="Whether to load pre-trained model")
    parser.add_argument("--bs", "--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", "--learning_rate", type=float, default=0.01, help="Learning rate")
    return parser


def get_warmup_lr(step, warmup_steps, lr):
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    return lr


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    args = get_parser().parse_args()
    
    mode = args.mode
    seed = args.seed
    noise_level = 0.5  # or 0.25
    if_load_pt = bool(args.pt)
    lr = args.lr
    bs = args.bs
    seq_len = 1500

    # Reproducibility
    np.random.seed(seed + 1234)
    random.seed(seed + 1234)
    torch.manual_seed(seed + 1234)
    torch.cuda.manual_seed_all(seed + 1234)

    # Output paths
    model_folder = f'ckpt/{mode}_{noise_level}'
    os.makedirs(model_folder, exist_ok=True)
    snapshot_name = f'seq_{seq_len}_seed_{seed}_bs_{bs}'

    # Data loading
    import pandas as pd
    train_df = pd.read_csv('data/Berlin_train_test.csv')
    train_df = train_df[train_df['train_or_test'] == 'train']
    file_list = train_df['file'].tolist()

    raw_train, y, mean, std = raw_data_statistics(file_list, seq_len, noise_level)

    train_idxs, val_idxs = train_test_split(np.arange(len(file_list)), test_size=0.2, random_state=seed)
    train_dataset = Loader_CO2M(raw_train, y, train_idxs, mask_ratio=0.3)
    val_dataset = Loader_CO2M(raw_train, y, val_idxs, mask_ratio=0.3)
    
    train_data_loader = DataLoader(train_dataset, batch_size=bs, num_workers=1, shuffle=True, pin_memory=False, drop_last=True)
    val_data_loader = DataLoader(val_dataset, batch_size=bs, num_workers=1, shuffle=False, pin_memory=False, drop_last=True)

    # Model initialization
    model = MXPT_Transformer(
        seq_len=seq_len, 
        xco2_channels=3, 
        pos_channels=4, 
        hidden=128, 
        n_layers=4, 
        attn_heads=8, 
        out_channel=7, 
        dropout=0.2, 
        mode=mode
    )
    model = model.cuda()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99))
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    model = torch.nn.DataParallel(model)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[1, 3, 6, 10, 18, 30, 50, 70, 100, 130, 170, 190], gamma=0.5)

    # Loss functions
    rec_loss_fn = masked_loss(mean=mean, std=std, batch=True)
    # PlumeRegLoss: a normalization parameter for measuring plume intensity regression
    plmreg_loss_fn = PlumeRegLoss(y_min=0.2, y_max=1.0)

    # Load pre-trained model if requested
    pt_path = f'ckpt/Pre-train_{noise_level}/seq_{seq_len}_seed_{seed}_bs_8'
    if if_load_pt:
        model = load_pt_model(model, pt_path)

    best_score = 1e6
    for epoch in range(200):
        iterator = tqdm(train_data_loader)
        total_loss_meter = AverageMeter()

        model.train()
        for step, sample in enumerate(iterator):
            if mode == 'Pre-train':
                # XCO2 data with partial masking
                x = sample["xco2_data"].cuda(non_blocking=True)
                # Includes all features to be reconstructed
                gt_rec = sample["gt_rec"].cuda(non_blocking=True)
                # Positions that are masked and included in rec_loss calculation
                masked = sample["masked"].cuda(non_blocking=True)
                
                out, _ = model(x, masked)  # (batch, seq_len, dimension)
                loss = rec_loss_fn(out, gt_rec, masked)

            elif mode == 'PlumeCLS':
                # Complete XCO2 observation sequence
                x = sample["gt_rec"].cuda(non_blocking=True)
                # Note: true_conc (gt_y) is not normalized and remains original raw values
                true_conc = sample["gt_y"].cuda(non_blocking=True)
                _, out = model(x, None) 
                loss = plmreg_loss_fn(out, true_conc)

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 0.999)
            optimizer.step()

            total_loss_meter.update(loss.item())
            iterator.set_description(
                f"Epoch: {epoch}; LR: {scheduler.get_last_lr()[-1]:.7f}; Loss: {total_loss_meter.val:.4f} ({total_loss_meter.avg:.4f})"
            )
            
            # Linear warmup for first 100 steps
            if epoch < 10:  
                for param_group in optimizer.param_groups:
                    param_group['lr'] = get_warmup_lr(step, warmup_steps=100, lr=scheduler.get_last_lr()[-1])

        scheduler.step()
        torch.cuda.empty_cache()
        
        if mode == 'Pre-train':
            score_recon = validate_recon(val_data_loader, model, std, mean)
            # Lower score indicates better performance
            if score_recon < best_score:
                best_score = score_recon
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'stage': mode,
                    'best_score': score_recon,
                }, os.path.join(model_folder, snapshot_name))
            print(f"Epoch: {epoch}, Reconstruction Score: {score_recon:.4f}, Best Score: {best_score:.4f}")

        elif mode == 'PlumeCLS':
            score = validate_plume_cls(val_data_loader, model, y_min=0.2, y_max=1.0)
            if score < best_score:
                best_score = score
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'stage': mode,
                    'best_score': best_score,
                }, os.path.join(model_folder, snapshot_name))
            print(f"Epoch: {epoch}, Plume Reg Score: {score:.4f}, Best Score: {best_score:.4f}")
