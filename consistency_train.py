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
from utils.losses import TverskyLoss, focal_loss, omni_comprehensive_loss, DiceLoss, ConsistencyLoss
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
    parser.add_argument("--re_train_weight_c", default=None, 
                        help="load retrain another weight") 
    parser.add_argument("--cell_model", type=str, required=True, 
                        choices=['RUnet++', 'AttUnet', 'Consis'],
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
        wandb.log({"train/train_iou_c": log['avg_iou_c'], "step": log['step']})
        wandb.log({"train/train_border_iou": log['avg_border_iou'], "step": log['step']})
        wandb.log({"train/train_border_iou_c": log['avg_border_iou_c'], "step": log['step']})
        wandb.log({"lr": log['lr'], "step": log['step']})
        wandb.log({"train/Image": wandb.Image(log['image']), "step": log['step'],
                   "train/Mask":{
                        "pred": wandb.Image(log['pred']),
                        "pred_c": wandb.Image(log['pred_c']),
                        "gt": wandb.Image(log['gt']), 
                    }, "step": log['step']
                  })

    elif mode =='val':
        wandb.log({"val/loss": log['avg_loss'], "step": log['step']})
        wandb.log({"val/val_iou": log['avg_iou'], "step": log['step']})
        wandb.log({"val/val_iou_c": log['avg_iou_c'], "step": log['step']})
        wandb.log({"val/val_border_iou": log['avg_border_iou'], "step": log['step']})
        wandb.log({"val/val_border_iou_c": log['avg_border_iou_c'], "step": log['step']})
        wandb.log({"val/Image": wandb.Image(log['image']), "step": log['step'],
                   "val/Mask":{
                        "pred": wandb.Image(log['pred']),
                        "pred_c": wandb.Image(log['pred_c']),
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


def forward_step(model, model_c, batch, loss_functions, device):
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
    masks_pred_c, border_pred_c = model_c(img)
    from torchvision.utils import save_image
    # print(masks_pred.shape[0])
    save_image(masks_pred, 'masks_pred.png')
    save_image(masks_pred_c, 'masks_pred_c.png')
    # for iter in range(masks_pred.shape[0]):
    #     save_image(masks_pred[iter], f'masks_pred[{iter}].png')
    #     save_image(masks_pred_c[iter], f'masks_pred_c[{iter}].png')
    # print("ddddd")

    masks_pred = torch.sigmoid(masks_pred)
    masks_pred_c = torch.sigmoid(masks_pred_c)
    border_pred = torch.sigmoid(border_pred)
    border_pred_c = torch.sigmoid(border_pred_c)

    Loss_BCE = loss_functions['bce']
    loss_bce = Loss_BCE(masks_pred, masks_pred)

    Loss_dice = loss_functions['dice']
    loss_dice = Loss_dice(masks_pred, gt_mask)

    Loss_Focal = loss_functions['focal']
    loss_focal = Loss_Focal(masks_pred, gt_mask)
    
    Loss_Omni = loss_functions['omni']
    loss_omni = Loss_Omni(masks_pred, gt_mask)
    
    # Loss_Consis = loss_functions['consis']
    # loss_consis = Loss_Consis(masks_pred, masks_pred)

    # print("loss_dice", loss_dice)
    # print("loss_focal", loss_focal)
    # print("loss_omni", loss_omni)
    # print("loss_consis", loss_bce)
    # loss = loss_dice + loss_focal + loss_omni 
    loss = loss_dice + loss_focal + loss_omni + loss_bce
    # print(loss_tversky_region.requires_grad)
    # print(loss_consis.requires_grad)

    if not np.isnan(IOU(gt_mask, masks_pred).detach().cpu().numpy()):
        iou = IOU(gt_mask, masks_pred)
    else:
        iou = 0
    if not np.isnan(IOU(gt_border, border_pred).detach().cpu().numpy()):
        border_iou = IOU(gt_border, border_pred)
    else:
        border_iou = 0
    if not np.isnan(IOU(gt_mask, masks_pred_c).detach().cpu().numpy()):
        iou_c = IOU(gt_mask, masks_pred_c)
    else:
        iou_c = 0
    if not np.isnan(IOU(gt_border, border_pred_c).detach().cpu().numpy()):
        border_iou_c = IOU(gt_border, border_pred_c)
    else:
        border_iou_c = 0

    log_images = {
        'image': img[0],
        'pred': torch.unsqueeze(masks_pred[:,0], dim=1)[0],
        'pred_c': torch.unsqueeze(masks_pred_c[:,0], dim=1)[0],
        'gt': gt_mask[0]
    }
    log_iou = {
        'iou': iou,
        'iou_c': iou_c,
        'border_iou': border_iou,
        'border_iou_c': border_iou_c
    }

    return loss, log_images, log_iou


def save_state_dict(is_multi, model_to_save, path):
    if is_multi:
        torch.save({'model_state_dict': model_to_save.module.state_dict()}, path)
    else:
        torch.save({'model_state_dict': model_to_save.state_dict()}, path)


if __name__ == "__main__":
    # Get Arguments
    args = get_args()
    # 1. Setup Model============================================================
    if args.cell_model == "Consis":
        model = ResUnetPlusPlus(channel=3)
        model_c = AttU_Net(img_ch=3, output_ch=args.num_class)
    else:
        print("Model Loading Error!")
        exit()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device_c = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print("Model Loading Success!")

    if args.re_train_weight != None:
        checkpoint = torch.load(args.re_train_weight , map_location="cuda:0")
        checkpoint_c = torch.load(args.re_train_weight_c , map_location="cuda:0")

        # Add additional dropout layer which is not including in the pretrained weight
        # Remove dropout layers from the pretrained_dict
        # pretrained_dict = {k: v for k, v in checkpoint.items() if 'dropout' not in k}

        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except:
            model.load_state_dict({k.replace('module.', ''): v 
                            for k, v in checkpoint['model_state_dict'].items()})
        print("Weight model Loading Success!")    
        
        try:
            model_c.load_state_dict(checkpoint_c['model_state_dict'])
        except:
            model_c.load_state_dict({k.replace('module.', ''): v 
                            for k, v in checkpoint_c['model_state_dict'].items()})
        print("Weight model_c Loading Success!")    

    # Multi-GPU
    if torch.cuda.device_count() > 1:
        print("Using Multiple GPU", torch.cuda.device_count())
        model = nn.DataParallel(model)
        model_c = nn.DataParallel(model_c)

    model.to(device)
    model_c.to(device)
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
    Loss_Dice = DiceLoss().to(device)
    Loss_Consistency = ConsistencyLoss().to(device)
    
    loss_functions = {
        'bce': Loss_BCE,
        'tversky_r':Loss_Tversky_R,
        'tversky_b':Loss_Tversky_B,
        'focal':Loss_focal,
        'omni':Loss_omni,
        'dice':Loss_Dice,
        'consis':Loss_Consistency
    }
    # ==========================================================================

    # 5. Forword Step===========================================================
    torch.cuda.empty_cache()

    for epoch_idx in range(args.max_epoch):
        train_iou_sum = []
        train_iou_c_sum = []
        train_border_iou_sum = []
        train_border_iou_c_sum = []
        train_loss_sum = []

        print(f'-----Epoch: {epoch_idx}-----')
        print("-----lr:", scheduler.optimizer.param_groups[0]['lr'])
        for batch_idx, batch in tqdm(enumerate(train_dataloader), 
                                     total=len(train_dataloader)):
            model.train()
            model_c.train()
            optimizer.zero_grad()
            
            train_loss, log_images, log_iou = forward_step(model, 
                                                           model_c, 
                                                           batch, 
                                                           loss_functions,
                                                           device)
            torch.autograd.set_detect_anomaly(True)
            train_loss.backward()

            optimizer.step()

            train_loss_sum += [train_loss.detach().cpu().numpy().item()]
            train_iou_sum += [log_iou['iou'].detach().cpu().numpy()]
            train_iou_c_sum += [log_iou['iou_c'].detach().cpu().numpy()]
            train_border_iou_sum += [log_iou['border_iou'].detach().cpu().numpy()]
            train_border_iou_c_sum += [log_iou['border_iou_c'].detach().cpu().numpy()]

        scheduler.step()
        if args.enable_wandb:
            wandb_log(wandb, mode='train', log={
                "step": epoch_idx,
                "avg_loss": sum(train_loss_sum) / len(train_loss_sum),
                "avg_iou": sum(train_iou_sum) / len(train_iou_sum),
                "avg_iou_c": sum(train_iou_c_sum) / len(train_iou_c_sum),
                "avg_border_iou": sum(train_border_iou_sum) / len(train_border_iou_sum),
                "avg_border_iou_c": sum(train_border_iou_c_sum) / len(train_border_iou_c_sum),
                "lr": optimizer.param_groups[0]['lr'],
                "image": reverse_normalized_image(log_images['image'].detach().cpu().numpy()),
                "pred":reverse_normalized_image(log_images['pred'].detach().cpu().numpy()),
                "pred_c":reverse_normalized_image(log_images['pred_c'].detach().cpu().numpy()),
                "gt":reverse_normalized_image(log_images['gt'].detach().cpu().numpy())
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
        print(f"train_avg_iou_c: {sum(train_iou_c_sum) / len(train_iou_c_sum)}")
        print(f"train_avg_border_iou: {sum(train_border_iou_sum) / len(train_border_iou_sum)}")
        print(f"train_avg_border_iou_c: {sum(train_border_iou_c_sum) / len(train_border_iou_c_sum)}")

        val_loss_sum = []
        val_iou_sum = []
        val_iou_c_sum = []
        val_border_iou_sum = []
        val_border_iou_c_sum = []

        for batch in tqdm(val_dataloader, total=len(val_dataloader)):
            model.eval()
            with torch.no_grad():
                val_loss, log_images, log_iou = forward_step(model, 
                                                             model_c,
                                                             batch, 
                                                             loss_functions, 
                                                             device)

            val_loss_sum += [val_loss.detach().cpu().numpy().item()]
            val_iou_sum += [log_iou['iou'].detach().cpu().numpy()]
            val_iou_c_sum += [log_iou['iou_c'].detach().cpu().numpy()]
            val_border_iou_sum += [log_iou['border_iou'].detach().cpu().numpy()]
            val_border_iou_c_sum += [log_iou['border_iou_c'].detach().cpu().numpy()]

        if args.enable_wandb:
            wandb_log(wandb, mode='val', log={
                "step": epoch_idx,
                "avg_loss": sum(val_loss_sum) / len(val_loss_sum),
                "avg_iou": sum(val_iou_sum) / len(val_iou_sum),
                "avg_iou_c": sum(val_iou_c_sum) / len(val_iou_c_sum),
                "avg_border_iou": sum(val_border_iou_sum) / len(val_border_iou_sum),
                "avg_border_iou_c": sum(val_border_iou_c_sum) / len(val_border_iou_c_sum),
                "image": reverse_normalized_image(log_images['image'].detach().cpu().numpy()),
                "pred":reverse_normalized_image(log_images['pred'].detach().cpu().numpy()),
                "pred_c":reverse_normalized_image(log_images['pred_c'].detach().cpu().numpy()),
                "gt":reverse_normalized_image(log_images['gt'].detach().cpu().numpy())
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
        print(f"valid_avg_iou_c: {sum(val_iou_c_sum) / len(val_iou_c_sum)}")
        print(f"valid_avg_border_iou: {sum(val_border_iou_sum) / len(val_border_iou_sum)}") 
        print(f"valid_avg_border_iou_c: {sum(val_border_iou_c_sum) / len(val_border_iou_c_sum)}") 

        if epoch_idx == 0:
            os.makedirs(args.ckpt_dir + "/" + args.run_name + "/Attunet/", exist_ok=True)
            os.makedirs(args.ckpt_dir + "/" + args.run_name + "/Resunet/", exist_ok=True)
            save_state_dict(torch.cuda.device_count() > 1, model_c,
                            f"{args.ckpt_dir}/{args.run_name}/Attunet/best_model.pt")
            save_state_dict(torch.cuda.device_count() > 1, model, 
                            f"{args.ckpt_dir}/{args.run_name}/Resunet/best_model.pt")
            print("---------- SAVE INITIAL WEIGHT-----------")
            global_val_iou = sum(val_iou_sum) / len(val_iou_sum)
            global_val_iou_c = sum(val_iou_c_sum) / len(val_iou_c_sum)

        if sum(val_iou_sum) / len(val_iou_sum) > global_val_iou:
            save_state_dict(torch.cuda.device_count() > 1, model, 
                            f"{args.ckpt_dir}/{args.run_name}/Resunet/best_model.pt")
            global_val_iou = sum(val_iou_sum) / len(val_iou_sum)
            print("------------ SAVE BEST Resunet WEIGHT------------")

        if sum(val_iou_c_sum) / len(val_iou_c_sum) > global_val_iou_c:
            save_state_dict(torch.cuda.device_count() > 1, model_c, 
                            f"{args.ckpt_dir}/{args.run_name}/Attunet/best_model.pt")
            global_val_iou_c = sum(val_iou_c_sum) / len(val_iou_c_sum)
            print("------------ SAVE BEST Attunet WEIGHT------------")

        print("global_val_iou:", global_val_iou)
        print("global_val_iou_c:", global_val_iou_c)

        save_state_dict(torch.cuda.device_count() > 1, model, 
                        f"{args.ckpt_dir}/{args.run_name}/Attunet/epoch_{epoch_idx}.pt")
        save_state_dict(torch.cuda.device_count() > 1, model_c, 
                        f"{args.ckpt_dir}/{args.run_name}/Resunet/epoch_{epoch_idx}.pt")
        print(f"---------- SAVE EPOCH_{epoch_idx} WEIGHT-----------")