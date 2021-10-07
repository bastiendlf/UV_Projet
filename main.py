from matplotlib import pyplot as plt
import cv2
from utils.customJpegCompressor import compress, decompress
import os

data_folder = './data/'
pictures = os.listdir(data_folder)


def show_image(img):
    plt.figure(figsize=(15, 15))
    plt.imshow(img, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()


img = cv2.imread(data_folder + pictures[0], 0)
show_image(img)

compressed_output = compress(img)
res_img = decompress(compressed_output, img.shape[1])

show_image(res_img)

artefacts = img - res_img
show_image(artefacts)
