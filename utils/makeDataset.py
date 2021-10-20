import numpy as np
import pickle
import cv2
import os
import glob
import utils.customJpegCompressor as JpegCompressor
from utils.blockJpeg import block_jpeg, get_quantification_matrix, floor, halfup, trunc, round_op, d1, d2, d3

INPUT_DATA_FOLDER = '../data/'
OUTPUT_DATA_FOLDER = '../output/datasets/'


def settings_to_dict(d, s, sh, sv, Q, fq, f2, fe, fo, fs):
    """
    Convert compression settings into a python dict (for easier manipulation)
    :param d: 1D-DCT function (d1, d2 or d3)
    :param s: scalar for 2D-descaling
    :param sh: 1x8 vector DCT divisors for row de-scaling
    :param sv: 1x8 vector DCT divisors for column de-scaling
    :param Q: quantization quality from 1 to 100
    :param fq: rounding operator after quantization
    :param f2: rounding operator after quantization with power of 2
    :param fe: rounding operator for even DCT coefficients in 1D-DCT de-scaling
    :param fo: rounding operator for odd DCT coefficients in 1D-DCT de-scaling
    :param fs: rounding operator for 2D-DCT de-scaling
    :return: python dict with the corresponding values
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
    :param Q: quantization quality from 1 to 100
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
    path_settings = os.path.join(OUTPUT_DATA_FOLDER, str(dataset_number), 'settings.pickle')
    if os.path.exists(path_settings):
        with open(path_settings, 'rb') as handle:
            settings = pickle.load(handle)
        return settings
    else:
        return None


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


def make_dataset_random_d_and_roundings(picture_folder, Qf):
    """
    Create an off-line dataset of jpeg compressed images with Q90, dct_function and random roundings functions
    :param picture_folder: input folder containing pictures to compress
    :param Qf: quantization values between 1 and 100
    :return: none
    """
    folder_name = f"Q{Qf}"
    pictures = [f for f in os.listdir(picture_folder) if f.endswith('.tif')]

    # fixed parameters
    Q_matrix = get_quantification_matrix(Qf)
    s = 2 ** 0
    sh = np.array([2 ** 11] * 8)
    sv = np.array([2 ** 15] * 8)

    # random parameters
    roundings = [floor, halfup, trunc, round_op]
    d_functions = [d1, d2, d3]

    # Make sure the output folder exits
    if not os.path.exists(OUTPUT_DATA_FOLDER):
        os.mkdir(OUTPUT_DATA_FOLDER)

    # Place pictures with same compression settings in a common folder
    dataset_output = os.path.join(OUTPUT_DATA_FOLDER, folder_name)
    if not os.path.exists(dataset_output):
        os.mkdir(dataset_output)

    already_computed_pictures = os.listdir(dataset_output)
    pictures_to_convert = [pic for pic in pictures if not (pic + ".npy" in already_computed_pictures)]
    nb_pic_to_convert = len(pictures_to_convert)

    for i, pic_name in enumerate(pictures_to_convert):
        print(f"{pic_name} | {100 * i / nb_pic_to_convert}%")
        img = cv2.imread(INPUT_DATA_FOLDER + pic_name, 0)

        # set random parameters
        d = np.random.choice(d_functions)
        fq = np.random.choice(roundings)
        f2 = np.random.choice(roundings)
        fe = np.random.choice(roundings)
        fo = np.random.choice(roundings)
        fs = np.random.choice(roundings)

        # Compute JPEG compression
        sliced_img = JpegCompressor.slice_image(img)
        DCToutput = [block_jpeg(block, d, s, sh, sv, Q_matrix, fq, f2, fe, fo, fs) for block in sliced_img]

        np.save(dataset_output + "/" + pic_name + ".npy", JpegCompressor.get_flat_average_dct_image(DCToutput))


def make_dataset_random_roundings(picture_folder, dct_function_name, Qf):
    """
    Create an off-line dataset of jpeg compressed images with Q90, dct_function and random roundings functions
    :param picture_folder: input folder containing pictures to compress
    :param dct_function_name: d1 d2 or d3
    :param Qf: quantization value (from 1 to 100)
    :return: none
    """
    DCT_FUNCTION = {
        "d1": d1,
        "d2": d2,
        "d3": d3
    }
    d = DCT_FUNCTION[dct_function_name]
    dataset_folder_name = f"{dct_function_name}_Q{Qf}"

    pictures = [f for f in os.listdir(picture_folder) if f.endswith('.tif')]

    # fixed parameters
    Q_matrix = get_quantification_matrix(Qf)
    s = 2 ** 0
    sh = np.array([2 ** 11] * 8)
    sv = np.array([2 ** 15] * 8)

    # random parameters
    roundings = [floor, halfup, trunc, round_op]

    # Make sure the output folder exits
    if not os.path.exists(OUTPUT_DATA_FOLDER):
        os.mkdir(OUTPUT_DATA_FOLDER)

    # Place pictures with same compression settings in a common folder
    dataset_output = os.path.join(OUTPUT_DATA_FOLDER, dataset_folder_name)
    if not os.path.exists(dataset_output):
        os.mkdir(dataset_output)

    already_computed_pictures = os.listdir(dataset_output)
    pictures_to_convert = [pic for pic in pictures if not (pic + ".npy" in already_computed_pictures)]
    nb_pic_to_convert = len(pictures_to_convert)

    for i, pic_name in enumerate(pictures_to_convert):
        print(f"{pic_name} | {100 * i / nb_pic_to_convert}%")
        img = cv2.imread(INPUT_DATA_FOLDER + pic_name, 0)

        # set random roundings
        fq = np.random.choice(roundings)
        f2 = np.random.choice(roundings)
        fe = np.random.choice(roundings)
        fo = np.random.choice(roundings)
        fs = np.random.choice(roundings)

        # Compute JPEG compression
        sliced_img = JpegCompressor.slice_image(img)
        DCToutput = [block_jpeg(block, d, s, sh, sv, Q_matrix, fq, f2, fe, fo, fs) for block in sliced_img]

        np.save(dataset_output + "/" + pic_name + ".npy", JpegCompressor.get_flat_average_dct_image(DCToutput))


def make_dataset_validation(picture_folder, dataset_name, Q_list, d_list, rounding_list):
    """
    Create an off-line dataset of jpeg compressed images with random settings
    :param rounding_list: roundings function
    :param d_list: 1D dct function
    :param Q_list: quantization values
    :param picture_folder: input folder containing pictures to compress
    :param dataset_name: str
    :return: none
    """

    pictures = [f for f in os.listdir(picture_folder) if f.endswith('.tif')]

    s = 2 ** 0
    sh = np.array([2 ** 11] * 8)
    sv = np.array([2 ** 15] * 8)

    # Make sure the output folder exits
    if not os.path.exists(OUTPUT_DATA_FOLDER):
        os.mkdir(OUTPUT_DATA_FOLDER)

    dataset_output = os.path.join(OUTPUT_DATA_FOLDER, str(dataset_name))
    if not os.path.exists(dataset_output):
        os.mkdir(dataset_output)

    # Load previously computed images ground truth
    if os.path.exists(os.path.join(dataset_output, "settings.pickle")):
        ground_truth = load_settings(dataset_name)
    else:
        ground_truth = []

    already_computed_pictures = os.listdir(dataset_output)
    pictures_to_convert = [pic for pic in pictures if not (pic + ".npy" in already_computed_pictures)]
    nb_pic_to_convert = len(pictures_to_convert)

    for i, pic_name in enumerate(pictures_to_convert):
        print(f"{pic_name} | {100 * i / nb_pic_to_convert}%")
        img = cv2.imread(INPUT_DATA_FOLDER + pic_name, 0)

        d = np.random.choice(d_list)
        Qf = np.random.choice(Q_list)
        Q_matrix = get_quantification_matrix(Qf)
        fq = np.random.choice(rounding_list)
        f2 = np.random.choice(rounding_list)
        fe = np.random.choice(rounding_list)
        fo = np.random.choice(rounding_list)
        fs = np.random.choice(rounding_list)

        # Compute JPEG compression
        sliced_img = JpegCompressor.slice_image(img)
        DCToutput = [block_jpeg(block, d, s, sh, sv, Q_matrix, fq, f2, fe, fo, fs) for block in sliced_img]

        np.save(dataset_output + "/" + pic_name + ".npy", JpegCompressor.get_flat_average_dct_image(DCToutput))

        ground_truth.append({
            "pic_name": pic_name,
            "d": d,
            "Q": Qf,
            "fq": fq,
            "f2": f2,
            "fe": fe,
            "fo": fo,
            "fs": fs
        })
    # Save ground truth
    with open(os.path.join(dataset_output, "settings.pickle"), 'wb') as handle:
        pickle.dump(ground_truth, handle, protocol=pickle.HIGHEST_PROTOCOL)


def make_dataset(picture_folder, dataset_number, d, s, sh, sv, Q, fq, f2, fe, fo, fs):
    """
    Create an off-line dataset of jpeg compressed images with the given settings passed
    :param picture_folder: input folder containing pictures to compress
    :param dataset_number: int
    :param d: 1D-DCT function (d1, d2 or d3)
    :param s: scalar for 2D-descaling
    :param sh: 1x8 vector DCT divisors for row de-scaling
    :param sv: 1x8 vector DCT divisors for column de-scaling
    :param Q: quantization quality from 1 to 100
    :param fq: rounding operator after quantization
    :param f2: rounding operator after quantization with power of 2
    :param fe: rounding operator for even DCT coefficients in 1D-DCT de-scaling
    :param fo: rounding operator for odd DCT coefficients in 1D-DCT de-scaling
    :param fs: rounding operator for 2D-DCT de-scaling
    :return: none
    """

    pictures = [f for f in os.listdir(picture_folder) if f.endswith('.tif')]
    Q_matrix = get_quantification_matrix(Q)

    for pic_name in pictures:
        print(pic_name)
        img = cv2.imread(INPUT_DATA_FOLDER + pic_name, 0)

        # Compute JPEG compression
        sliced_img = JpegCompressor.slice_image(img)
        DCToutput = [block_jpeg(block, d, s, sh, sv, Q_matrix, fq, f2, fe, fo, fs) for block in sliced_img]

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
        return {}, []
    else:
        settings = load_settings(dataset_number)
        dct_images_name = glob.glob(os.path.join(OUTPUT_DATA_FOLDER, str(dataset_number), '*.npy'))
        return settings, [np.load(curr) for curr in dct_images_name]
