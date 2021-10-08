import numpy as np
import cv2
import utils.roundings as rnd
import utils.blockJpeg as blockJpeg

BLOCK_SIZE = 8


def select_q_matrix(q_name):
    q10 = np.array([[80, 60, 50, 80, 120, 200, 255, 255],
                    [55, 60, 70, 95, 130, 255, 255, 255],
                    [70, 65, 80, 120, 200, 255, 255, 255],
                    [70, 85, 110, 145, 255, 255, 255, 255],
                    [90, 110, 185, 255, 255, 255, 255, 255],
                    [120, 175, 255, 255, 255, 255, 255, 255],
                    [245, 255, 255, 255, 255, 255, 255, 255],
                    [255, 255, 255, 255, 255, 255, 255, 255]])

    q50 = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                    [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 130, 99]])

    q90 = np.array([[3, 2, 2, 3, 5, 8, 10, 12],
                    [2, 2, 3, 4, 5, 12, 12, 11],
                    [3, 3, 3, 5, 8, 11, 14, 11],
                    [3, 3, 4, 6, 10, 17, 16, 12],
                    [4, 4, 7, 11, 14, 22, 21, 15],
                    [5, 7, 11, 13, 16, 12, 23, 18],
                    [10, 13, 16, 17, 21, 24, 24, 21],
                    [14, 18, 19, 20, 22, 20, 20, 20]])
    if q_name == "Q10":
        return q10
    elif q_name == "Q50":
        return q50
    elif q_name == "Q90":
        return q90
    else:
        return np.ones((8, 8))


def compress_decompress(img, q_name='Q10', rounding=rnd.round):
    compressed = compress(img, q_name, rounding)
    decompressed = decompress(compressed, img.shape[1], q_name, rounding)
    return decompressed


def compress(img, q_name='Q10', rounding=rnd.round):
    """
    Take a gray scale image (width and height should be multiple of 8) and convert it into a jpeg compressed object.
    :param img: Should be gray scale image
    :param q_name: quantification matrix name
    :param rounding: rounding type
    :return: jpeg compressed image (without entropy coding)
    """
    img = np.float32(img) - 128
    sliced = slice_image(img)
    DCToutput = compute_dct(sliced)
    quantiOutput = quantification(DCToutput, select_q_matrix(q_name), rounding)

    return quantiOutput


def decompress(compressed_output, img_pxl_width, q_name='Q10', rounding=rnd.round):
    inverse_quantiOutput = inverse_quantification(compressed_output, select_q_matrix(q_name), rounding)
    idct_output = compute_idct(inverse_quantiOutput)
    unsclice = unslice_image(idct_output, img_pxl_width)
    img_res = unsclice + 128

    return img_res


def slice_image(image):
    height, width = image.shape
    sliced = []  # new list for 8x8 sliced image
    # dividing 8x8 parts
    currY = 0  # current Y index
    for i in range(BLOCK_SIZE, height + 1, BLOCK_SIZE):
        currX = 0  # current X index
        for j in range(BLOCK_SIZE, width + 1, BLOCK_SIZE):
            sliced.append(image[currY:i, currX:j])
            currX = j
        currY = i
    return sliced


def compute_dct(sliced_img):
    dct_output = [cv2.dct(part) for part in sliced_img]
    return dct_output


def compute_dct_blockjpeg(sliced_img, d, s, sh, sv, Q, fq, f2, fe, fo, fs):
    dct_output = [blockJpeg.block_jpeg(part, d, s, sh, sv, Q, fq, f2, fe, fo, fs) for part in sliced_img]
    return dct_output


def quantification(dct_output, selected_qmatrix, rounding=rnd.round):
    quantification_output = []
    for block_dct in dct_output:
        quantification_output.append(rounding(block_dct / selected_qmatrix))
    return quantification_output


def inverse_quantification(quantiOutput, selected_qmatrix, rounding=rnd.round):
    inverse_quantiOutput = []
    for block_dct in quantiOutput:
        inverse_quantiOutput.append(rounding(block_dct * selected_qmatrix))
    return inverse_quantiOutput


def compute_idct(inverse_quantiOutput):
    invList = [cv2.idct(ipart) for ipart in inverse_quantiOutput]
    return invList


def unslice_image(idct_output, img_width):
    row = 0
    rowNcol = []
    for j in range(int(img_width / BLOCK_SIZE), len(idct_output) + 1, int(img_width / BLOCK_SIZE)):
        rowNcol.append(np.hstack((idct_output[row:j])))
        row = j
    res = np.vstack((rowNcol))
    return res
