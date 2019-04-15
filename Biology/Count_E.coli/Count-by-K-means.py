# -*- coding: utf-8 -*-
import cv2
import copy
import numpy as np
from sklearn.cluster import KMeans
import os
import argparse

if __name__ == "__main__":
    tp = lambda x:list(map(int, x.split('.')))
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path',  type=str, default='img/E.coil.png', help='specify the path to the input image.')
    parser.add_argument('--output_path', type=str, default='img/{}-means-{}-E.coil.png', help='specify the path to the output image.')
    parser.add_argument('--K',           type=int, default=15, help="It is the 'K' used in 'K-means.'")
    parser.add_argument('--RGB',         type=tp,  default="222.219.203", help="Input E_coil's color. You can know that by 'Digital Color Meter.' (ex: 222.219.203)")
    args = parser.parse_args()

    img  = cv2.imread(args.input_path)
    img_shape = img.shape

    x = img.reshape(-1, 3).astype(float)
    model = KMeans(n_clusters = args.K)
    model.fit(x)

    cls = model.predict(x)
    mu_colors = np.array([]).reshape(-1,3)
    for k in range(args.K):
        mu_colors = np.vstack([mu_colors, np.average(x[cls == k], axis = 0)])

    R,G,B = args.RGB
    BGR = [B,G,R]
    k_e = np.argmin(np.sum(abs((mu_colors - BGR) * 2), axis=1))
    for k in range(args.K):
        if k == k_e:
            x[cls == k] = [255,255,255]
        else:
            x[cls == k] = [0,0,0]

    binary_img = x.astype(int).reshape(img_shape)
    cv2.imwrite("img/{}-means-tmp-E.coil.png".format(args.K), binary_img)

    img = cv2.imread("img/{}-means-tmp-E.coil.png".format(args.K))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nLabels, labelImage = cv2.connectedComponents(gray)

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

    cv2.imwrite(args.output_path.format(args.K, nLabels), dst)
    print("image was saved at '{}'".format(os.path.join(os.getcwd(), args.output_path.format(args.K, nLabels))))
