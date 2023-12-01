import numpy as np
import torch.nn as nn
import torch
from torch.nn import functional as F

class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.5, beta=0.5, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

        return 1 - Tversky


class ActiveContourLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ActiveContourLoss, self).__init__()

    def forward(self, inputs, targets, lambdaP=1., mu=1., eps=1e-8):
        x = targets[:, :, 1:, :] - targets[:, :, :-1, :]  # horizontal and vertical directions
        y = targets[:, :, :, 1:] - targets[:, :, :, :-1]

        delta_x = x[:, :, 1:, :-2] ** 2
        delta_y = y[:, :, :-2, 1:] ** 2
        delta_u = (delta_x + delta_y).abs()

        lenth = (delta_u + eps).sqrt().sum(-1)  # equ.(11) in the paper

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        region_in = targets * ((inputs - 1) ** 2)
        region_out = (1 - targets) * ((inputs) ** 2)
        region_l = mu * region_in + region_out

        region_l = region_l.sum(-1) * lambdaP
        return region_l.mean() + lenth.mean()
    

class focal_loss(nn.Module):
    def __init__(self, alpha=0.25,  gamma=2, weight=None, reduction='mean'):
        super(focal_loss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, input, target):
        # Calculate cross-entropy loss
        # ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        Loss_BCE = nn.BCEWithLogitsLoss(weight=self.weight, reduction='none')
        loss_bce = Loss_BCE(input, target)

        # Calculate the modulating factor
        p_t = torch.exp(-loss_bce)
        focal_loss = (1 - p_t) ** self.gamma * loss_bce

        # Apply weight if provided
        if self.weight is not None:
            focal_loss = self.weight * focal_loss
            
        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_loss = alpha_t * focal_loss
            
        #############################################################################
        # p = torch.sigmoid(input)
        # ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction="none")
        # p_t = p * target + (1 - p) * (1 - target)
        # loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        # if self.alpha >= 0:
        #     alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        #     loss = alpha_t * loss
        #     focal_loss = alpha_t * focal_loss
        #############################################################################

        # print("focal:", focal_loss)
        # print("losss:", loss)

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'none':
            return focal_loss
        else:
            raise ValueError("Invalid reduction option. Use 'mean', 'sum', or 'none'.")
        

# Normalized Cross Correlation 
def NCC(inputs, targets):
    c = (inputs - inputs.mean()) / (inputs.std() * len(inputs))
    d = (targets - targets.mean()) / (targets.std())
    c = c.cpu().detach().numpy()
    d = d.cpu().detach().numpy()
    ncc = np.correlate(c, d, 'valid')
    return ncc.mean ()

# Calculation of Tversky Index
def TverskyIndex(inputs, targets, adding_term=1, beta=0.1): 
    TP = (inputs * targets).sum()
    FP = ((1 - targets) * inputs).sum()
    FN = (targets * (1 - inputs)).sum()
    
    Tversky = (TP + adding_term) / (TP + beta*FP+ (1-beta)*FN + adding_term)
    return Tversky.mean()

class omni_comprehensive_loss(nn.Module):
    def __init__(self):
        super (omni_comprehensive_loss, self).__init__()
    def forward(self, inputs, targets):
        alpha = 0.5
        #sigmoid activation layer
        # inputs = F.sigmoid (inputs)
        #flatten label (target) and prediction (input) tensors.
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        #compute binary cross-entropy
        BCE = F.binary_cross_entropy (inputs, targets, reduction='mean')
        omni_comprehensive = (1 - (alpha * NCC(inputs, targets) +\
                             (1-alpha) * TverskyIndex(inputs, targets))) * BCE
        return omni_comprehensive


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        # Flatten the predictions and targets
        input_flat = input.view(-1)
        target_flat = target.view(-1)

        # Calculate the intersection and union
        intersection = torch.sum(input_flat * target_flat)
        union = torch.sum(input_flat) + torch.sum(target_flat)

        # Calculate the Dice coefficient
        dice_coefficient = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Calculate the Dice Loss
        loss = 1.0 - dice_coefficient

        return loss

# ConsistencyLoss MSE ver.
# Small loss value, about 0.01 ~ 0.001
# May proper to big batch size above 16 
# class ConsistencyLoss(nn.Module):
#     def __init__(self, consistency_criterion=nn.MSELoss(), temperature=1.0):
#         super(ConsistencyLoss, self).__init__()
#         self.consistency_criterion = consistency_criterion
#         self.temperature = temperature

#     def forward(self, predictions1, predictions2):
#         # Apply temperature scaling to soften the predictions

#         from torchvision.utils import save_image
#         save_image(predictions1, 'predictions1.png')
#         save_image(predictions2, 'predictions2.png')

#         # Calculate the consistency loss using the specified criterion
#         loss = self.consistency_criterion(predictions1, predictions2)

#         return loss
    
# ConsistencyLoss K.L. divergence ver.
class ConsistencyLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ConsistencyLoss, self).__init__()
        self.temperature = temperature

    def forward(self, predictions1, predictions2):
        # Apply temperature scaling to soften the predictions
        from torchvision.utils import save_image

        save_image(predictions1, 'predictions1.png')
        save_image(predictions2, 'predictions2.png')

        # print(predictions1.min(), predictions1.max())

        # predictions1 = F.log_softmax(predictions1 / self.temperature, dim=1)
        # predictions2 = F.log_softmax(predictions2 / self.temperature, dim=1)

        # print(predictions1.min(), predictions1.max())

        # save_image(predictions1, 'predictions11.png')
        # save_image(predictions2, 'predictions22.png')

        # Calculate KL Divergence
        # kl_div_loss = F.kl_div(predictions1.log(), predictions2, reduction='batchmean') + \
        #               F.kl_div(predictions2.log(), predictions1, reduction='batchmean')
        kl_loss  = nn.KLDivLoss(reduction="mean", log_target=True)
        kl_div_loss = kl_loss(predictions1.log(), predictions2.log())

        return kl_div_loss
    
# ConsistencyLoss Cross-entropy ver.
# class ConsistencyLoss(nn.Module):
#     def __init__(self, temperature=1.0):
#         super(ConsistencyLoss, self).__init__()
#         self.temperature = temperature

#     def forward(self, predictions1, predictions2):
#         # Apply temperature scaling to soften the predictions
#         predictions1 = F.softmax(predictions1 / self.temperature, dim=1)
#         predictions2 = F.softmax(predictions2 / self.temperature, dim=1)

#         # Calculate cross-entropy loss
#         ce_loss = -torch.mean(torch.sum(predictions1 * torch.log(predictions2), dim=1))

#         return ce_loss
