import numpy as np
import os
from utils.blockJpeg import floor, halfup, trunc, round_op, d1, d2, d3
import utils.makeDataset as DatasetMaker

OUTPUT_DATA_FOLDER = '../output/datasets/'
INPUT_DATA_FOLDER = '../data/'

Q_list = [10, 50, 90]
d_list = [d1, d2, d3]
rounding_list = [floor, halfup, trunc, round_op]

if __name__ == '__main__':
    if os.getcwd().endswith("/Code"):
        os.chdir(os.path.join(os.getcwd(), "utils"))
    print(f"Computing validation dataset ...")
    DatasetMaker.make_dataset_validation(INPUT_DATA_FOLDER, "validation", Q_list, d_list, rounding_list)
