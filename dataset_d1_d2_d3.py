import utils.makeDataset as DatasetMaker
import os
import sys

INPUT_DATA_FOLDER = '../data/'

d_list = ["d1", "d2", "d3"]
Qf = 90

if __name__ == '__main__':
    if os.getcwd().endswith("/Code"):
        os.chdir(os.path.join(os.getcwd(), "utils"))

    # If there is an arg passed while launching program
    if len(sys.argv) - 1 != 0 and str(sys.argv[1]) in d_list:
        d_value = str(sys.argv[1])
        print(f"Computing one dataset with Qf={Qf}, d={d_value} and random roundings ...")
        DatasetMaker.make_dataset_random_roundings(INPUT_DATA_FOLDER, d_value, Qf)

    else:  # else compute all datasets
        print(f"Computing all datasets with Qf={Qf} and random roundings ...")
        for d in d_list:
            print(f"Computing dataset with dct function {d} ...")
            DatasetMaker.make_dataset_random_roundings(INPUT_DATA_FOLDER, d, Qf)
