import argparse
import pickle
import glob
import numpy as np
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case_txt', type=str, required=True)
    parser.add_argument('--data_dir_path', type=str, required=True)
    parser.add_argument('--pickle_dir', type=str, required=True)
    parser.add_argument('--split_ratio', type=float, required=True)
    return parser.parse_args()


def read_case_list_txt(txt_path):
    case_list = []
    try:
        with open(txt_path, 'r') as f:
            for n in f:
                case_list.append(n.strip())
        return case_list
    except:
        print('[ERROR] Read file not found' + txt_path)

    return case_list

if __name__ == "__main__":
    args = get_args()
    work_root_path = os.path.join('/work', os.listdir('/work')[0])
    case_list = read_case_list_txt(args.case_txt)
    data_dir = args.data_dir_path
    image_path_list = []

    if not os.path.exists(args.pickle_dir):
        os.mkdir(args.pickle_dir)

    for case in case_list:
        for image_file in glob.glob(f'{data_dir}/img/{case}/*.png'):
            patch_name = os.path.basename(image_file).split('.')[0]
            label_path = f'{data_dir}/label/{case}/{patch_name}.png'
            image_path_list.append([image_file, label_path]) 

    # Shuffle and Split the Data List into Train and Valid list
    image_path_list = np.array(image_path_list)
    np.random.shuffle(image_path_list)
    total_len = np.shape(image_path_list)[0]
    train_ratio = args.split_ratio
    train_num = int(total_len * train_ratio)

    # Pickle Saving Path
    train_data_pkl_path = f"{args.pickle_dir}/train.pkl"
    val_data_pkl_path = f"{args.pickle_dir}/val.pkl"

    with open(train_data_pkl_path, 'wb') as f:
        pickle.dump(image_path_list[:train_num,:], f)

    with open(val_data_pkl_path, 'wb') as f:
        pickle.dump(image_path_list[train_num:,:], f)

    print("Pickle Files Save Successfully!")