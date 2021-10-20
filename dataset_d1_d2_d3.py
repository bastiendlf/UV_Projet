import utils.makeDataset as DatasetMaker
import os

OUTPUT_DATA_FOLDER = '../output/datasets/'
INPUT_DATA_FOLDER = '../data/'

d_list = ["d1", "d2", "d3"]
Qf = 10

if __name__ == '__main__':
    if os.getcwd().endswith("/Code"):
        os.chdir(os.path.join(os.getcwd(), "utils"))

    d = d_list[2]

    print(f"Computing dataset with dct function {d} ...")
    DatasetMaker.make_dataset_random_roundings(INPUT_DATA_FOLDER, d, Qf)

