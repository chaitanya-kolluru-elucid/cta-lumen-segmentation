import torch
import torch.nn as nn
import torch.nn.functional as F
from soft_skeleton import soft_skel
from monai.networks import one_hot


class soft_cldice(nn.Module):
    def __init__(self, iter_=3, smooth = 1.):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        skel_pred = soft_skel(y_pred, self.iters)
        skel_true = soft_skel(y_true, self.iters)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:,1:,...])+self.smooth)/(torch.sum(skel_pred[:,1:,...])+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:,1:,...])+self.smooth)/(torch.sum(skel_true[:,1:,...])+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice


def soft_dice(y_true, y_pred):
    """[function to compute dice loss]

    Args:
        y_true ([float32]): [ground truth image]
        y_pred ([float32]): [predicted image]

    Returns:
        [float32]: [loss value]
    """
    # Include background in our dice calculations

    smooth = 1e-5
    intersection = torch.sum((y_true * y_pred)[:,1:,...])
    coeff = (2. *  intersection + smooth) / (torch.sum(y_true[:,1:,...]) + torch.sum(y_pred[:,1:,...]) + smooth)
    return (1. - coeff)


class soft_dice_cldice_ce(nn.Module):
    def __init__(self, iter_=3, dice_weight = 0.5, cldice_weight = 0.5, smooth = 1e-5, num_classes = 3 , lumen_class=1): #TODO: Changing smooth parameter and weighting
        super(soft_dice_cldice_ce, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.dice_weight = dice_weight
        self.cldice_weight = cldice_weight
        self.num_classes = num_classes
        self.lumen_class = lumen_class

    def forward(self, y_pred, y_true):

        # Change the target to one hot embeddings
        target = one_hot(y_true, num_classes=self.num_classes) 

        # Binarize the predictions and calculate the Dice coefficient
        y_pred = torch.softmax(y_pred, 1)
        dice = soft_dice(target, y_pred)

        # For clDice, we are only interested in the lumen class (index 1 should be 1 to get the lumen out)
        skel_pred = soft_skel(y_pred[:,self.lumen_class,...], self.iter)
        skel_true = soft_skel(target[:,self.lumen_class,...], self.iter)
        tprec = (torch.sum(torch.multiply(skel_pred, target[:,self.lumen_class,...]))+self.smooth)/(torch.sum(skel_pred)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred[:,self.lumen_class,...]))+self.smooth)/(torch.sum(skel_true)+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)

        return self.dice_weight*dice + self.cldice_weight*cl_dice