# #!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BinaryDiceLoss(nn.Module):
    """
    Dice loss for binary segmentation
    
    Args:
        smooth: A float number to smooth loss and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        reduction: Reduction method to apply, return mean over batch if 'mean',
                   return sum if 'sum', return a tensor of shape [N,] if 'none'
    
    Input:
        predict: A tensor of shape [N, 1, H, W] with values in [0, 1] (after sigmoid)
        target: A tensor of shape [N, 1, H, W] with binary values {0, 1}
    
    Returns:
        Loss tensor according to arg reduction
    
    Note:
        This loss expects predictions AFTER sigmoid activation.
        The model should output probabilities directly.
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        
        # Flatten spatial dimensions
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        # Calculate Dice coefficient
        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice Loss
    
    Args:
        bce_weight: Weight for BCE loss, default: 0.5
        dice_weight: Weight for Dice loss, default: 0.5
        smooth: Smoothing parameter for Dice loss
    
    Input:
        predict: A tensor of shape [N, 1, H, W] with values in [0, 1] (after sigmoid)
        target: A tensor of shape [N, 1, H, W] with binary values {0, 1}
    
    Returns:
        Combined loss
    
    Note:
        This loss expects predictions AFTER sigmoid activation.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCELoss()
        self.dice_loss = BinaryDiceLoss(smooth=smooth)
    
    def forward(self, predict, target):
        bce = self.bce_loss(predict, target)
        dice = self.dice_loss(predict, target)
        return self.bce_weight * bce + self.dice_weight * dice


# Utility function for one-hot encoding (kept for potential future use)
def make_one_hot(input, num_classes):
    """
    Convert class index tensor to one hot encoding tensor.
    
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)
    return result