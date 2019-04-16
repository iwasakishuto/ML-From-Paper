# -*- coding: utf-8 -*-
# -*- OpenCV -*-
import cv2
import copy
import numpy as np
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path',  type=str, default='img/E.coil.png', help='specify the path to the input image.')
    parser.add_argument('--output_path', type=str, default='img/th-{}-E.coil.png', help='specify the path to the output image.')
    parser.add_argument('--Block_size',  type=int, default=9, help='Please enter odd number. It decides the size of neighbourhood area.')
    parser.add_argument('--C',           type=int, default=2, help='It is just a constant which is subtracted from the mean or weighted mean calculated.')
    args = parser.parse_args()

    img  = cv2.imread(args.input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th   = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, args.Block_size, args.C)
    nLabels, labelImage = cv2.connectedComponents(th)

    print("Number of E.coils: {}".format(nLabels))

    colors = []
    for i in range(nLabels):
        colors.append(np.random.randint(0, 255,3))

    height, width, channels = img.shape
    dst = copy.copy(img)
    for y in range(height):
        for x in range(width):
            if labelImage[y, x] > 0:
                dst[y, x] = colors[labelImage[y, x]]
            else:
                dst[y, x] = [0, 0, 0]

    cv2.imwrite(args.output_path.format(nLabels), dst)
    print("image was saved at '{}'".format(os.path.join(os.getcwd(), args.output_path.format(nLabels))))
