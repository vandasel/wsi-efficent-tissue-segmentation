import torch.nn as nn
import torch
from monai.networks.utils import one_hot
from monai.losses import DiceLoss, TverskyLoss 


class CustomDiceCELoss(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=255, label_smoothing=0.02)
        self.dice = DiceLoss(
            include_background=True,
            to_onehot_y=False,                   
	        softmax=False            
        )
        self.num_classes = num_classes

    def forward(self, preds, targets): 
        loss_ce = self.ce(preds, targets)
        mask = (targets != 255).float().unsqueeze(1) 
        targets_clean = torch.where(targets == 255, torch.zeros_like(targets), targets)
        targets_oh = one_hot(targets_clean.unsqueeze(1), num_classes=self.num_classes)
        
        targets_oh_masked = targets_oh * mask
        
        preds_softmax = torch.softmax(preds, dim=1)
        preds_masked = preds_softmax * mask

        loss_dice = self.dice(preds_masked, targets_oh_masked)

        return loss_ce + loss_dice


class CustomTverskyCELoss(nn.Module):
    def __init__(self, num_classes=3, alpha=0.3, beta=0.7):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=255, label_smoothing=0.02)
        self.tversky = TverskyLoss(
            include_background=True, 
            to_onehot_y=False,                   
            softmax=False,
            alpha=alpha,
            beta=beta
        )
        self.num_classes = num_classes

    def forward(self, preds, targets): 
        loss_ce = self.ce(preds, targets)
        
        mask = (targets != 255).float().unsqueeze(1) 
        targets_clean = torch.where(targets == 255, torch.zeros_like(targets), targets)
        targets_oh = one_hot(targets_clean.unsqueeze(1), num_classes=self.num_classes)
        
        targets_oh_masked = targets_oh * mask
        
        preds_softmax = torch.softmax(preds, dim=1)
        preds_masked = preds_softmax * mask

        loss_tversky = self.tversky(preds_masked, targets_oh_masked)

        return loss_ce + loss_tversky