import utils.makeDataset as DatasetMaker
import os

OUTPUT_DATA_FOLDER = '../output/datasets/'
INPUT_DATA_FOLDER = '../data/'

Q_list = [10, 50, 90]

if __name__ == '__main__':
    if os.getcwd().endswith("/Code"):
        os.chdir(os.path.join(os.getcwd(), "utils"))

    for Q in Q_list:
        print(f"Computing dataset with Qf = {Q} ...")
        DatasetMaker.make_dataset_random_d_and_roundings(INPUT_DATA_FOLDER, Q)

