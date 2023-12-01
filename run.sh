username="kevin20307"
version="v1_0_0_1128_pretrained_consis+dice+focal+omni"
dataset=NCKU
# model_version="Immune_Atten_Unet"
# model_version="Immune_Cell_ResUnet_PlusPlus"
model_version="Immune_Cell_Dual"

# build pickle file
# python save_training_pkl.py  --case_txt "case_list.txt" \
#                              --data_dir_path "/work/$username/Immune_Cell/train/training_data" \
#                              --split_ratio 0.8 \
#                              --pickle_dir "/work/$username/Immune_Cell/train/pkl/$dataset/$version"

# # show pickle file
# python show_pkl.py  --pkl_path "/work/$username/Immune_Cell/train/pkl/$dataset/$version/train.pkl"
# python show_pkl.py  --pkl_path "/work/$username/Immune_Cell/train/pkl/$dataset/$version/val.pkl"

# training
# python train.py --pkl_dir "/work/$username/Immune_Cell/train/pkl/$dataset/$version/" \
#                 --project "$model_version" \
#                 --run "$version" \
#                 --ckpt_dir "/work/$username/Immune_Cell/train/weights/$model_version" \
#                 --num_class 1 \
#                 --max_epoch 1000 \
#                 --train_bz 16 \
#                 --val_bz 8 \
#                 --lr 0.001 \
#                 --cell_model RUnet++ \
#                 --re_train_weight "/work/kevin20307/Immune_Cell/train/weights/Immune_Cell_ResUnet_PlusPlus/v1_0_0_1125_lym_pretrained_focal+omni_test/best_model.pt"
#                 # --re_train_weight "/work/$username/Immune_Cell/train/weights/Immune_Cell_Atten_Unet/v1_0_0_0707_Atten_Retrain/best_model.pt"
#                 # --re_train_weight "/work/$username/Immune_Cell/train/weights/$model_version/v1_0_0_1123_Atten_Pannuke_nuclei/best_model.pt"
#                 # --cell_model AttUnet 
#                 # --enable_wandb True \

# Dual training
python consistency_train.py --pkl_dir "/work/$username/Immune_Cell/train/pkl/$dataset/$version/" \
                            --project "$model_version" \
                            --run "$version" \
                            --ckpt_dir "/work/$username/Immune_Cell/train/weights/$model_version" \
                            --num_class 1 \
                            --max_epoch 1000 \
                            --train_bz 8 \
                            --val_bz 8 \
                            --lr 0.001 \
                            --cell_model Consis \
                            --re_train_weight "/work/kevin20307/Immune_Cell/train/weights/Immune_Cell_ResUnet_PlusPlus/v1_0_0_1127_lym_pretrained_dice+focal+omni/best_model.pt" \
                            --re_train_weight_c "/work/kevin20307/Immune_Cell/train/weights/Immune_Atten_Unet/v1_0_0_1124_Atten_Pannuke_pretrained_lym_nAUG/best_model.pt" \
                            # --enable_wandb True \
