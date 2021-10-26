import numpy as np
import cv2
import utils.roundings as rnd
import utils.blockJpeg as blockJpeg

BLOCK_SIZE = 8


def get_flat_average_dct_image(dct_image):
    """
    Takes a list of 8x8 DCT blocks for an image and compute the average DCT block and then flatten it
    :param dct_image: list of 8x8 DCT blocks
    :return: list of 64 average DCT coefficients
    """
    average_dct = np.mean(np.array(dct_image), axis=0)
    return average_dct.flatten()


def compress_decompress(img, quantification_quality=50, rounding=rnd.round_op):
    """
    Jpeg encodes an image and decode it
    :param img: Should be gray scale image
    :param quantification_quality: number between 1 and 100
    :param rounding: rounding function
    :return: decompressed gray scale image
    """
    compressed = compress(img, quantification_quality, rounding)
    decompressed = decompress(compressed, img.shape[1], quantification_quality, rounding)
    return decompressed


def compress(img, quantification_quality=50, rounding=rnd.round_op):
    """
    Take a gray scale image (width and height should be multiple of 8) and convert it into a jpeg compressed object.
    :param img: Should be gray scale image
    :param quantification_quality: quantification matrix quality value (from 1 to 100)
    :param rounding: rounding type
    :return: jpeg compressed image (without entropy coding)
    """
    img = np.float32(img) - 128
    sliced = slice_image(img)
    DCToutput = compute_dct(sliced)
    quantiOutput = quantification(DCToutput, blockJpeg.get_quantification_matrix(quantification_quality), rounding)

    return quantiOutput


def decompress(compressed_output, img_pxl_width, quantification_quality=50, rounding=rnd.round_op):
    """
    Convert a list of DCT into a gray scale image
    :param compressed_output: jpeg compressed image (without entropy coding)
    :param img_pxl_width: int
    :param quantification_quality: int between 1 and 100
    :param rounding: rounding function
    :return: gray scale image
    """
    inverse_quantiOutput = inverse_quantification(compressed_output,
                                                  blockJpeg.get_quantification_matrix(quantification_quality),
                                                  rounding)
    idct_output = compute_idct(inverse_quantiOutput)
    unsclice = unslice_image(idct_output, img_pxl_width)
    img_res = unsclice + 128

    return img_res


def slice_image(image):
    """
    Slice a 2D array into a list of 8x8 blocks
    :param image: 2D gray scale array
    :return: list of 8x8 arrays
    """
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
    """
    Convert an image into a list of DCT (computed with OpenCv)
    :param sliced_img: list of 8x8 pixels blocks
    :return: list of 8x8 dct output
    """
    dct_output = [cv2.dct(part) for part in sliced_img]
    return dct_output


def compute_dct_blockjpeg(sliced_img, d, s, sh, sv, Q, fq, f2, fe, fo, fs):
    """
    Compute 2D-DCT, quantization + rounding for an 8x8 pixels block
    :param sliced_img: list of 8x8 pixels block
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
    :return: list of 8x8 2D-DCT output
    """
    dct_output = [blockJpeg.block_jpeg(part, d, s, sh, sv, Q, fq, f2, fe, fo, fs) for part in sliced_img]
    return dct_output


def quantification(dct_output, selected_qmatrix, rounding=rnd.round_op):
    """
    Quantization of the DCT output
    :param dct_output: list of 8x8 dct blocks
    :param selected_qmatrix: 8x8 quantization matrix
    :param rounding: rounding function
    :return: list of quantified and rounded dct matrices
    """
    quantification_output = []
    for block_dct in dct_output:
        quantification_output.append(rounding(block_dct / selected_qmatrix))
    return quantification_output


def inverse_quantification(quantiOutput, selected_qmatrix, rounding=rnd.round_op):
    """
    Remove quantization on a list of DCT blocks quantified
    :param quantiOutput: list of quantified and rounded dct matrices
    :param selected_qmatrix: 8x8 quantization matrix
    :param rounding: rounding function
    :return: list of 8x8 dct blocks unquantified
    """
    inverse_quantiOutput = []
    for block_dct in quantiOutput:
        inverse_quantiOutput.append(rounding(block_dct * selected_qmatrix))
    return inverse_quantiOutput


def compute_idct(inverse_quantiOutput):
    """
    Compute the inverse DCT
    :param inverse_quantiOutput: list of 8x8 dct blocks unquantified
    :return: list of 8x8 pixel blocks
    """
    invList = [cv2.idct(ipart) for ipart in inverse_quantiOutput]
    return invList


def unslice_image(idct_output, img_width):
    """
    convert a list of 8x8 pixels blocks into a 2D image in gray scale
    :param idct_output: list of 8x8 pixel blocks
    :param img_width: int
    :return: built image
    """
    row = 0
    rowNcol = []
    for j in range(int(img_width / BLOCK_SIZE), len(idct_output) + 1, int(img_width / BLOCK_SIZE)):
        rowNcol.append(np.hstack((idct_output[row:j])))
        row = j
    res = np.vstack((rowNcol))
    return res
