import torch
import torch.nn as nn
import torch.nn.functional as F


class masked_loss(nn.Module):
    """
    Custom L1 loss for masked reconstruction tasks.
    """
    def __init__(self, mean, std, batch=True):
        super(masked_loss, self).__init__()
        self.mean = mean
        self.std = std
        self.batch = batch
        self.loss_fn = nn.L1Loss(reduction='mean')
    
    def forward(self, pred, target, masked):
        device = pred.device
        mean = self.mean.to(device)
        std = self.std.to(device)

        # Reconstruct values using mean and std
        # Note: Sigmoid is not used for regression tasks
        pred_unnorm = (pred * std + mean)
        target_unnorm = (target * std + mean)

        # Compute loss only on masked positions
        loss = self.loss_fn(pred_unnorm[masked == 1.0], target_unnorm[masked == 1.0])
        return loss 


class PlumeRegLoss(nn.Module):
    """
    L1 loss for Plume intensity regression with normalization.
    """
    def __init__(self, y_min=0.20, y_max=1.0, y_thres=0.50):
        # Background < y_min, Plume > y_max 
        super(PlumeRegLoss, self).__init__()
        self.y_min = y_min
        self.y_max = y_max
        self.y_thres = y_thres
        self.loss_fn = nn.L1Loss(reduction='mean')

    def forward(self, pred, true_conc):
        """
        Forward pass for plume regression loss.
        """
        y_norm = self.normalize(true_conc)
        loss = self.loss_fn(pred, y_norm)
        return loss
    
    def normalize(self, true_conc):
        """
        Normalize raw concentration values to [0, 1] range.
        """
        y_norm = torch.where(
            true_conc <= self.y_min, 
            torch.zeros_like(true_conc),
            torch.where(
                true_conc >= self.y_max, 
                torch.ones_like(true_conc),
                (true_conc - self.y_min) / (self.y_max - self.y_min)
            )
        )
        
        # Ensure values are within [0, 1]
        y_norm = y_norm.unsqueeze(dim=-1)
        return y_norm
