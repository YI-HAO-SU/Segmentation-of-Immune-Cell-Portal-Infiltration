import argparse
import cv2
import os
import torch
import wandb
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import immune_cell_dataset
from models.atten_unet import AttU_Net
from models.res_unet_plus import ResUnetPlusPlus
from utils.losses import TverskyLoss, focal_loss, omni_comprehensive_loss, DiceLoss
from utils.metrics import IOU


def get_args():
    """
    Get arguments from command
    
    Args:
        None
    
    Return:
        args(dict of arguments): dictionary object with arguments.  
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_pkl_dir", "--pkl_dir", type=str, 
                        help="directory path of pikle file")
    parser.add_argument("--project_name", "--project", type=str, 
                        help="name of the wandb project name")
    parser.add_argument("--run_name", "--run", type=str, 
                        help="name of this run")
    parser.add_argument("--ckpt_dir", type=str, 
                        help="directory path of saving weight file")
    parser.add_argument("--log_images_dir", type=str, default=None, 
                        help="directory to save images during training (last batch first patch)")
    parser.add_argument("--num_class", type=int, 
                        help="number of class to feed in model")
    parser.add_argument("--max_epoch", "--eopch", type=int, 
                        help="number of maximum number of epoch")
    parser.add_argument("--train_batch_size", "--train_bz", type=int, 
                        help="training batch size")
    parser.add_argument("--val_batch_size", "--val_bz", type=int, 
                        help="validation batch size")
    parser.add_argument("--learning_rate", "--lr", type=float, 
                        help="number of learning rate")
    parser.add_argument("--enable_wandb", "--wandb", default=False, 
                        help="to user wandb or not for logging training")
    parser.add_argument("--re_train_weight", default=None, 
                        help="load retrain weight")
    parser.add_argument("--cell_model", type=str, required=True, 
                        choices=['RUnet++', 'AttUnet'],
                        help="immune cell segmentation model")

    return parser.parse_args()


def wandb_log(wandb, mode:str, log:dict):
    """
    Record value to wandb log dictionary

    Args:
        wandb(wandb object): main wandb object to set
        mode(str): indicate it's training mode or valid mode
        log(dict): contain value to be recorded to wandb
    
    Return:
        None
    """

    if mode == 'train':
        wandb.log({"train/totol loss": log['avg_loss'], "step": log['step']})
        wandb.log({"train/train_iou": log['avg_iou'], "step": log['step']})
        wandb.log({"train/train_border_iou": log['avg_border_iou'], "step": log['step']})
        wandb.log({"lr": log['lr'], "step": log['step']})
        wandb.log({"train/Image": wandb.Image(log['image']), "step": log['step'],
                   "train/Mask":{
                        "pred": wandb.Image(log['pred']),
                        "gt": wandb.Image(log['gt']), 
                    }, "step": log['step']
                  })

    elif mode =='val':
        wandb.log({"val/loss": log['avg_loss'], "step": log['step']})
        wandb.log({"val/val_iou": log['avg_iou'], "step": log['step']})
        wandb.log({"val/val_border_iou": log['avg_border_iou'], "step": log['step']})
        wandb.log({"val/Image": wandb.Image(log['image']), "step": log['step'],
                   "val/Mask":{
                        "pred": wandb.Image(log['pred']),
                        "gt": wandb.Image(log['gt']), 
                    }, "step": log['step']
                  })


def reverse_normalized_image(img):
    """
    Reverse normalized image to original image with value in 255.
    
    Args:
        img(numpy array): normalized image to be reversed.
    
    Return:
        image_return(numpy array): original image. 
    """
    max_value = img.max()
    image = img * 255 / max_value if max_value != 0 else img
    image = image.transpose(1, 2, 0)[:,:,::-1].copy()
    image_return = image.copy()

    return image_return


def forward_step(model, batch, loss_functions, device):
    """
    Network forward process
    
    Args:
        model(nn.module): network
        batch(tenor): batch data 
        loss_functions(dict of loss function object): contain loss functions
        device(torch.device object): designated device for training.

    Returns:
        loss(tensor): total training loss
        log_images(dict of images): images dictionary with origianl image of the
                                    first image in the batch.
        log_iou(dict of IOU value): dictionary with cell and border iou.
    """
    img, gt_mask, gt_border = batch
    img ,gt_mask, gt_border = img.to(device), \
                              gt_mask.to(device), \
                              gt_border.to(device)
    masks_pred, border_pred = model(img)

    masks_pred = torch.sigmoid(masks_pred)
    border_pred = torch.sigmoid(border_pred)
    # binary_masks_pred = torch.where(masks_pred > 0.5, 1.0, 0.0)
    # binary_border_pred = torch.where(border_pred > 0.5, 1.0, 0.0)

    # Loss_BCE = loss_functions['bce']
    # loss_bce = Loss_BCE(masks_pred, gt_mask)
    # loss_bce_border = Loss_BCE(border_pred, gt_border)
    # print(loss_bce)
    # print(loss_bce_border)

    Loss_Tversky_R = loss_functions['tversky_r']
    loss_tversky_region = Loss_Tversky_R(masks_pred, gt_mask)

    # Loss_Tversky_B = loss_functions['tversky_b']    
    # loss_tversky_border = Loss_Tversky_B(border_pred, gt_border)

    Loss_Focal = loss_functions['focal']
    loss_focal = Loss_Focal(masks_pred, gt_mask)
    
    Loss_Omni = loss_functions['omni']
    loss_omni = Loss_Omni(masks_pred, gt_mask)
    # print(loss_omni)
    
    # loss = loss_bce + loss_bce_border + loss_tversky_region + loss_tversky_border
    loss = loss_tversky_region + loss_focal + loss_omni
    # print(loss_tversky_region.requires_grad)
    # print(loss_tversky_border.requires_grad)

    if not np.isnan(IOU(gt_mask, masks_pred).detach().cpu().numpy()):
        iou = IOU(gt_mask, masks_pred)
    else:
        iou = 0
    if not np.isnan(IOU(gt_border, border_pred).detach().cpu().numpy()):
        border_iou = IOU(gt_border, border_pred)
    else:
        border_iou = 0

    log_images = {
        'image': img[0],
        'pred': torch.unsqueeze(masks_pred[:,0], dim=1)[0],
        'gt': gt_mask[0]
    }
    log_iou = {
        'iou': iou,
        'border_iou': border_iou
    }

    return loss, log_images, log_iou


def save_state_dict(is_multi, path):
    if is_multi:
        torch.save(
                {'model_state_dict': model.module.state_dict()}, path)
    else:
        torch.save(
                {'model_state_dict': model.state_dict()}, path)


if __name__ == "__main__":
    # Get Arguments
    args = get_args()
    # 1. Setup Model============================================================
    if args.cell_model == "RUnet++" and "ResUnet_PlusPlus" in args.ckpt_dir:
        model = ResUnetPlusPlus(channel=3)

    elif args.cell_model == "AttUnet" and "Atten_Unet" in args.ckpt_dir:
        model = AttU_Net(img_ch=3, output_ch=args.num_class)
    else:
        print("Model Loading Error!")
        exit()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print("Model Loading Success!")

    if args.re_train_weight != None:
        checkpoint = torch.load(args.re_train_weight , map_location="cuda:0")

        # Add additional dropout layer which is not including in the pretrained weight
        # Remove dropout layers from the pretrained_dict
        pretrained_dict = {k: v for k, v in checkpoint.items() if 'dropout' not in k}

        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except:
            print("e")
            model.load_state_dict({k.replace('module.', ''): v 
                            for k, v in checkpoint['model_state_dict'].items()})
        print("Weight Loading Success!")    

    if torch.cuda.device_count() > 1:
        print("Using Multiple GPU", torch.cuda.device_count())
        model = nn.DataParallel(model)
    model.to(device)
    # ==========================================================================

    # 2. Initialize wandb======================================================= 
    if args.enable_wandb:
        wandb = wandb
        wandb.init(project=args.project_name)
        wandb.run.name = args.run_name
        wandb.config.max_epoch = args.max_epoch
        wandb.config.train_batch_size = args.train_batch_size
        wandb.config.val_batch_size = args.val_batch_size
        wandb.config.learning_rate = args.learning_rate
    # ==========================================================================
    
    # 3. Setup Dataloader=======================================================
    train_transform = transforms.Compose([
            # transforms.RandomVerticalFlip(0.5),
            # transforms.RandomHorizontalFlip(0.5),
            transforms.ToPILImage(),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2,
                                       saturation=0.2,
                                       hue=0.2,
                                       contrast=0.2)], p=0.5),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])
    val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])

    train_dataset = immune_cell_dataset(args.data_pkl_dir, 
                                        train_transform, 
                                        'train')
    valid_dataset = immune_cell_dataset(args.data_pkl_dir, 
                                        val_transform, 
                                        'val')
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=args.train_batch_size,
                                  pin_memory=False, 
                                  num_workers=4, 
                                  shuffle=True, 
                                  drop_last=False)
    val_dataloader = DataLoader(valid_dataset, 
                                batch_size=args.val_batch_size, 
                                pin_memory=False, 
                                num_workers=4, 
                                shuffle=False, 
                                drop_last=False)
    print("Data Loading Success!")
    # ==========================================================================

    # 4. Setup Loss and Optimizer===============================================
    global_val_loss = 0
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,60,100], gamma=0.3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                                    10, 5)

    Loss_BCE = nn.BCEWithLogitsLoss().to(device)
    Loss_Tversky_R = TverskyLoss(0.5, 0.5).to(device)  # similar to dice
    Loss_Tversky_B = TverskyLoss(0.1, 0.9).to(device)  # 
    Loss_focal = focal_loss().to(device)
    Loss_omni = omni_comprehensive_loss().to(device)
    
    loss_functions = {
        'bce': Loss_BCE,
        'tversky_r':Loss_Tversky_R,
        'tversky_b':Loss_Tversky_B,
        'focal':Loss_focal,
        'omni':Loss_omni
    }
    # ==========================================================================

    # 5. Forword Step===========================================================
    torch.cuda.empty_cache()

    for epoch_idx in range(args.max_epoch):
        train_iou_sum = []
        train_border_iou_sum = []
        train_loss_sum = []
        

        print(f'-----Epoch: {epoch_idx}-----')
        print("-----lr:", scheduler.optimizer.param_groups[0]['lr'])
        for batch_idx, batch in tqdm(enumerate(train_dataloader), 
                                     total=len(train_dataloader)):
            model.train()
            optimizer.zero_grad()
            
            train_loss, log_images, log_iou = forward_step(model, batch, 
                                                        loss_functions, device)
            train_loss.backward()

            optimizer.step()

            train_loss_sum += [train_loss.detach().cpu().numpy().item()]
            train_iou_sum += [log_iou['iou'].detach().cpu().numpy()]
            train_border_iou_sum += [log_iou['border_iou'].detach().cpu().numpy()]

        scheduler.step()
        if args.enable_wandb:
            wandb_log(wandb, mode='train', log={
                "step": epoch_idx,
                "avg_loss": sum(train_loss_sum) / len(train_loss_sum),
                "avg_iou": sum(train_iou_sum) / len(train_iou_sum),
                "avg_border_iou": \
                        sum(train_border_iou_sum) / len(train_border_iou_sum),
                "lr": optimizer.param_groups[0]['lr'],
                "image": reverse_normalized_image(
                                    log_images['image'].detach().cpu().numpy()),
                "pred":reverse_normalized_image(
                                    log_images['pred'].detach().cpu().numpy()),
                "gt":reverse_normalized_image(
                                    log_images['gt'].detach().cpu().numpy())
            })

        if args.log_images_dir != None:
            cv2.imwrite(
            f"{args.log_images_dir}/{args.run_name}/train_{epoch_idx}_pred.png", 
            reverse_normalized_image(log_images['image'].detach().cpu().numpy()))
            cv2.imwrite(
            f"{args.log_images_dir}/{args.run_name}/train_{epoch_idx}_img.png", 
            reverse_normalized_image(log_images['pred'].detach().cpu().numpy()))
            cv2.imwrite(
            f"{args.log_images_dir}/{args.run_name}/train_{epoch_idx}_gt.png", 
            reverse_normalized_image(log_images['gt'].detach().cpu().numpy()))
        
        print(f"train_avg_loss: {sum(train_loss_sum) / len(train_loss_sum)}")
        print(f"train_avg_iou: {sum(train_iou_sum) / len(train_iou_sum)}")
        print(f"train_avg_border_iou: \
                {sum(train_border_iou_sum) / len(train_border_iou_sum)}")

        val_loss_sum = []
        val_iou_sum = []
        val_border_iou_sum = []

        for batch in tqdm(val_dataloader, total=len(val_dataloader)):
            model.eval()
            with torch.no_grad():
                val_loss, log_images, log_iou = forward_step(model, 
                                                            batch, 
                                                            loss_functions, 
                                                            device)

            val_loss_sum += [val_loss.item()]
            val_iou_sum += [log_iou['iou']]
            val_border_iou_sum += [log_iou['border_iou']]

        if args.enable_wandb:
            wandb_log(wandb, mode='val', log={
                "step": epoch_idx,
                "avg_loss": sum(val_loss_sum) / len(val_loss_sum),
                "avg_iou": sum(val_iou_sum) / len(val_iou_sum),
                "avg_border_iou": \
                        sum(val_border_iou_sum) / len(val_border_iou_sum),
                "image": reverse_normalized_image(
                        log_images['image'].detach().cpu().numpy()),
                "pred":reverse_normalized_image(
                        log_images['pred'].detach().cpu().numpy()),
                "gt":reverse_normalized_image(
                        log_images['gt'].detach().cpu().numpy())
            })
        
        if args.log_images_dir != None:
            cv2.imwrite(
            f"{args.log_images_dir}/{args.run_name}/val_{epoch_idx}_img.png", 
            reverse_normalized_image(log_images['image'].detach().cpu().numpy()))            
            cv2.imwrite(
            f"{args.log_images_dir}/{args.run_name}/val_{epoch_idx}_gt.png", 
            reverse_normalized_image(log_images['pred'].detach().cpu().numpy()))
            cv2.imwrite(
            f"{args.log_images_dir}/{args.run_name}/val_{epoch_idx}_pred.png", 
            reverse_normalized_image(log_images['gt'].detach().cpu().numpy()))

        print(f"valid_avg_loss: {sum(val_loss_sum) / len(val_loss_sum)}")
        print(f"valid_avg_iou: {sum(val_iou_sum) / len(val_iou_sum)}")
        print(f"valid_avg_border_iou: \
                {sum(val_border_iou_sum) / len(val_border_iou_sum)}") 

        if epoch_idx == 0:
            os.makedirs(args.ckpt_dir + "/" + args.run_name + "/", exist_ok=True)
            save_state_dict(torch.cuda.device_count() > 1, 
                            f"{args.ckpt_dir}/{args.run_name}/best_model.pt")
            print("---------- SAVE INITIAL WEIGHT-----------")
            global_val_loss = sum(val_loss_sum) / len(val_loss_sum)

        elif sum(val_loss_sum) / len(val_loss_sum) < global_val_loss:
            save_state_dict(torch.cuda.device_count() > 1, 
                            f"{args.ckpt_dir}/{args.run_name}/best_model.pt")
            global_val_loss = sum(val_loss_sum) / len(val_loss_sum)
            print("------------ SAVE BEST WEIGHT------------")

        save_state_dict(torch.cuda.device_count() > 1, 
                        f"{args.ckpt_dir}/{args.run_name}/epoch_{epoch_idx}.pt")
        print(f"---------- SAVE EPOCH_{epoch_idx} WEIGHT-----------")