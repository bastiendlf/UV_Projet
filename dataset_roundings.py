import utils.makeDataset as DatasetMaker
import os
import sys
from utils.blockJpeg import floor, halfup, trunc, round_op, d1, d2, d3

INPUT_DATA_FOLDER = '../data/'

roundings = {
    "round": round_op,
    "floor": floor,
    "halfup": halfup,
    "trunc": trunc
}

Qf = 90
d = d3

if __name__ == '__main__':
    if os.getcwd().endswith("/Code"):
        os.chdir(os.path.join(os.getcwd(), "utils"))

    # If there is an arg passed while launching program
    if len(sys.argv) - 1 != 0 and str(sys.argv[1]) in roundings.keys():
        round_name = str(sys.argv[1])
        print(f"Computing one dataset with roundings={round_name}, Qf={Qf} and d={d} ...")
        DatasetMaker.make_dataset_same_roundings(INPUT_DATA_FOLDER, roundings[round_name], round_name, Qf, d)

    else:  # else compute all datasets
        print(f"Computing all datasets for all roundings with Qf={Qf} and d={d} ...")
        for key in roundings.keys():
            print(f"Computing dataset with rounding {key} ...")
            DatasetMaker.make_dataset_same_roundings(INPUT_DATA_FOLDER, roundings[key], key, Qf, d)
