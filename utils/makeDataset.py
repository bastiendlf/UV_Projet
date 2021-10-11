import numpy as np
import pickle
import cv2
import os
import glob
import utils.customJpegCompressor as JpegCompressor
from utils.blockJpeg import block_jpeg

INPUT_DATA_FOLDER = '../data/'
OUTPUT_DATA_FOLDER = '../output/datasets/'
max_pic = 5


def settings_to_dict(d, s, sh, sv, Q, fq, f2, fe, fo, fs):
    """
    Convert compression settings into a python dict (for easier manipulation)
    :param d: 1D-DCT function (d1, d2 or d3)
    :param s: scalar for 2D-descaling
    :param sh: 1x8 vector DCT divisors for row de-scaling
    :param sv: 1x8 vector DCT divisors for column de-scaling
    :param Q: 8x8 quantification matrix
    :param fq: rounding operator after quantization
    :param f2: rounding operator after quantization with power of 2
    :param fe: rounding operator for even DCT coefficients in 1D-DCT de-scaling
    :param fo: rounding operator for odd DCT coefficients in 1D-DCT de-scaling
    :param fs: rounding operator for 2D-DCT de-scaling
    :return: python dict with corresponding values
    """
    settings = {
        'd': d,
        's': s,
        'sh': sh,
        'sv': sv,
        'Q': Q,
        'fq': fq,
        'f2': f2,
        'fe': fe,
        'fo': fo,
        'fs': fs
    }
    return settings


def save_settings(output, d, s, sh, sv, Q, fq, f2, fe, fo, fs):
    """
    Save python dict with settings offline into pickle format
    :param output: path to save the settings
    :param d: 1D-DCT function (d1, d2 or d3)
    :param s: scalar for 2D-descaling
    :param sh: 1x8 vector DCT divisors for row de-scaling
    :param sv: 1x8 vector DCT divisors for column de-scaling
    :param Q: 8x8 quantification matrix
    :param fq: rounding operator after quantization
    :param f2: rounding operator after quantization with power of 2
    :param fe: rounding operator for even DCT coefficients in 1D-DCT de-scaling
    :param fo: rounding operator for odd DCT coefficients in 1D-DCT de-scaling
    :param fs: rounding operator for 2D-DCT de-scaling
    :return: none
    """
    settings = settings_to_dict(d, s, sh, sv, Q, fq, f2, fe, fo, fs)

    with open(os.path.join(output, "settings.pickle"), 'wb') as handle:
        pickle.dump(settings, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_settings(dataset_number):
    """
    Load Python dict containing jpeg compression settings.
    :param dataset_number: int or str
    :return: Python dict
    """
    with open(os.path.join(OUTPUT_DATA_FOLDER, str(dataset_number), 'settings.pickle'), 'rb') as handle:
        settings = pickle.load(handle)
    return settings


def make_dataset_dict_settings(picture_folder, dataset_number, settings):
    """
    Create an off-line dataset of jpeg compressed images with the given settings passed in the dict settings
    :param picture_folder: input folder containing pictures to compress
    :param dataset_number: int
    :param settings: python dict
    :return: none
    """
    d = settings['d']
    s = settings['s']
    sh = settings['sh']
    sv = settings['sv']
    Q = settings['Q']
    fq = settings['fq']
    f2 = settings['f2']
    fe = settings['fe']
    fo = settings['fo']
    fs = settings['fs']

    make_dataset(picture_folder, dataset_number, d, s, sh, sv, Q, fq, f2, fe, fo, fs)


def make_dataset(picture_folder, dataset_number, d, s, sh, sv, Q, fq, f2, fe, fo, fs):
    """
    Create an off-line dataset of jpeg compressed images with the given settings passed
    :param picture_folder: input folder containing pictures to compress
    :param dataset_number: int
    :param d: 1D-DCT function (d1, d2 or d3)
    :param s: scalar for 2D-descaling
    :param sh: 1x8 vector DCT divisors for row de-scaling
    :param sv: 1x8 vector DCT divisors for column de-scaling
    :param Q: 8x8 quantification matrix
    :param fq: rounding operator after quantization
    :param f2: rounding operator after quantization with power of 2
    :param fe: rounding operator for even DCT coefficients in 1D-DCT de-scaling
    :param fo: rounding operator for odd DCT coefficients in 1D-DCT de-scaling
    :param fs: rounding operator for 2D-DCT de-scaling
    :return:
    """
    pictures = os.listdir(picture_folder)
    for pic_name in pictures[:max_pic]:
        img = cv2.imread(INPUT_DATA_FOLDER + pic_name, 0)

        # Compute JPEG compression
        sliced_img = JpegCompressor.slice_image(img)
        DCToutput = [block_jpeg(block, d, s, sh, sv, Q, fq, f2, fe, fo, fs) for block in sliced_img]

        # Make sure the output folder exits
        if not os.path.exists(OUTPUT_DATA_FOLDER):
            os.mkdir(OUTPUT_DATA_FOLDER)

        # Place pictures with same compression settings in a common folder
        dataset_output = os.path.join(OUTPUT_DATA_FOLDER, str(dataset_number))
        if not os.path.exists(dataset_output):
            os.mkdir(dataset_output)

        # Save compression settings into pickle format (may be usefull latter)
        if not os.path.exists(os.path.join(dataset_output, 'settings.pickle')):
            save_settings(dataset_output, d, s, sh, sv, Q, fq, f2, fe, fo, fs)

        # Save the jpeg compressed image in the corresponding folder
        np.save(dataset_output + "/" + pic_name + ".npy", DCToutput)


def load_dataset(dataset_number):
    """
    Load an off_line already_computed dataset of jpeg compressed images
    :param dataset_number: int
    :return: tuple (dict_settings, list[compressed_images]
    """
    path_to_dataset = os.path.join(OUTPUT_DATA_FOLDER, str(dataset_number))
    if not os.path.exists(path_to_dataset):
        return []
    else:
        settings = load_settings(dataset_number)
        dct_images_name = glob.glob(os.path.join(OUTPUT_DATA_FOLDER, str(dataset_number), '*.npy'))
        return settings, [np.load(curr) for curr in dct_images_name]
