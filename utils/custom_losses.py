
import torch.nn as nn
import torch
from monai.networks.utils import one_hot
from monai.losses import DiceLoss

class CustomDiceCELoss(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # We can use crossentropy here so we ignore 255 and later we will use dice, only this works idk
        self.ce = nn.CrossEntropyLoss(ignore_index=255)
        self.dice = DiceLoss(to_onehot_y=False, softmax=False, include_background=True)
        self.num_classes = num_classes
   

    def forward(self, preds, targets):
        loss_ce = self.ce(preds, targets)
        mask = (targets != 255).unsqueeze(1).float() # unsqueeze cause we need to idu 
        targets_clean = torch.where(targets == 255, torch.zeros_like(targets), targets)
        targets_oh = one_hot(targets_clean.unsqueeze(1), num_classes=self.num_classes)

        preds_soft = torch.softmax(preds, dim=1)

        preds_masked = preds_soft * mask
        targets_masked = targets_oh * mask

        loss_dice = self.dice(preds_masked, targets_masked)

        return (loss_ce + loss_dice)  # weight for later testing
