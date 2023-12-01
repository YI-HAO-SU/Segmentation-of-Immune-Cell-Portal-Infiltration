import argparse
import pickle
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", type=str, help="pikle file saving path")
    args = parser.parse_args()

    return args
    
if __name__ == "__main__":
    args = get_args()
    data = []
    with open(args.pkl_path, 'rb') as f:
        try:
            data.extend(pickle.load(f))
        except Exception as e:
            print(e)
            
    print("Pickle File as Follows")
    print(f"Size of Pickle is \n{len(data)}")
    print(f"The First data is \n{data[0]}")