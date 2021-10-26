import utils.makeDataset as DatasetMaker
import os
import sys

INPUT_DATA_FOLDER = '../data/'

Q_list = [10, 50, 90]

if __name__ == '__main__':
    if os.getcwd().endswith("/Code"):
        os.chdir(os.path.join(os.getcwd(), "utils"))

    # If there is an arg passed while launching program
    if len(sys.argv) - 1 != 0 and str(sys.argv[1]) in Q_list:
        Q_value = str(sys.argv[1])
        print(f"Computing one dataset with Qf={Q_value} and random d and roundings ...")
        DatasetMaker.make_dataset_random_d_and_roundings(INPUT_DATA_FOLDER, Q_value)

    else:  # else compute all datasets
        print(f"Computing all datasets for all roundings with random d and roundings ...")
        for Q in Q_list:
            print(f"Computing dataset with Qf = {Q} ...")
            DatasetMaker.make_dataset_random_d_and_roundings(INPUT_DATA_FOLDER, Q)
