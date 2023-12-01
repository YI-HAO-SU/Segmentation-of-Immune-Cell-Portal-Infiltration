import torch

def IOU(mask, output):
    output = torch.round(output)
    intersection = torch.sum(torch.mul(mask, output))
    union = torch.sum(mask) + torch.sum(output) - intersection
    
    return intersection / union

def Precision(mask, output):
    output = torch.round(output)
    intersection = torch.sum(torch.mul(mask, output))
    return intersection / torch.sum(output)

def Recall(mask, output):
    output = torch.round(output)
    intersection = torch.sum(torch.mul(mask, output))
    return intersection / torch.sum(mask)

def Dice_cofficient(mask, output):
    output = torch.round(output)
    numerator = torch.sum(torch.mul(mask, output))
    denominator = torch.sum(mask ) + torch.sum(output)
    return numerator * 2 / denominator