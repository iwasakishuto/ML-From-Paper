# coding: utf-8
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array
from keras.layers.core import Lambda
from keras.models import Sequential
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def load_image(img_path):
    """
    関数の概要：VGG16のモデルに合うように画像を前処理して読み込む
    """
    img = load_img(img_path, target_size=(224, 224)) # PIL形式で画像を取得する。
    x = img_to_array(img)                            # ndarray の形式に整形する。
    x = np.expand_dims(x, axis=0)                    # 次元を追加する(batch_size の分)
    x = preprocess_input(x)                          # VGG16 が学習させた形に前処理を行う。
    return x

def grad_cam(model, image, category_index, layer_name="block5_conv3"):
    """
    関数の概要：GradCAM で可視化した画像を ndarray の形で返す。
              特徴マップごとの重要度(α)を求め、重み付き和を取ることでCNNを可視化する。
    @param input_model   ：予測に利用したモデル(VGG16)
    @param image         ：上記のモデルで予測したい画像データ(適当な形に前処理済み)
    @param category_index：予測ラベルのindex
    @param layer_name    ：利用する特徴量マップ
    """
    nb_classes = 1000 # VGG16の予測に利用した予測ラベルの総数
    conv_output =  [l for l in model.layers if l.name is layer_name][0].output # 利用する特徴マップの output.

    one_hot = K.one_hot([category_index], nb_classes) # 正解ラベルだけ1のone-hotベクトル。
    signal = tf.multiply(model.output, one_hot) # これによって、正解だと考えたラベルの影響(原因)のみを考える。
    loss = K.mean(signal) # ここに特に意味はない(と思う。)

    grads = tf.gradients(loss, conv_output)[0] # 勾配を求める。
    norm_grads = tf.divide(grads, K.sqrt(K.mean(K.square(grads))) + tf.constant(1e-5)) # テンソルをL2ノルムで正則化。

    # 入力に対して勾配を求めることのできる関数を用意。
    gradient_function = K.function([model.layers[0].input], [conv_output, norm_grads])

    output, grads_val = gradient_function([image]) # ともに、shape=(1, 14, 14, 512)
    output, grads_val = output[0, :], grads_val[0, :, :, :] # shapeの調整。shape=(14, 14, 512)

    weights = np.mean(grads_val, axis = (0, 1)) # 特徴マップごとに平均化することで、この値が各特徴マップの重要度(α)に相当。
    cam = np.ones(output.shape[:2], dtype = np.float32) # 画像サイズに合わせたベース画像を作成。

    for i, w in enumerate(weights):
        cam += w * output[:, :, i] # 各特徴マップを、その重要度に比率した重みを掛けて足し合わせる。

    cam = cv2.resize(cam, (224, 224)) # 元々の入力画像のサイズに整形。
    # cam = np.maximum(cam, 0) # 0以下の値を0にする。
    # heatmap = cam / np.max(cam) # 最大値で割ることで正規化。
    heatmap = (cam - np.min(cam)) / (np.max(cam) - np.min(cam)) # heatmap を 0-1正規化する。

    # 前処理された画像を、元のRGBの画像に戻す。
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * (cam / np.max(cam))
    return np.uint8(cam), heatmap

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='specify the path to the input image.')
    parser.add_argument('--output_path', type=str, default='./img/GradCAM.png', help='specify the path to the output image.')
    parser.add_argument('--heatmap_path', type=str, default='./img/Heatmap.png', help='specify the path to the heatmap.')
    args = parser.parse_args()

    model = VGG16(weights='imagenet') # ImageNet で学習させた重み付きの VGG16 モデルを読み込む。
    x = load_image(args.input_path)   # VGG16 に適した形に前処理し、画像を渡す。
    predictions = model.predict(x)    # 予測を行う。

    top_1 = decode_predictions(predictions)[0][0]   # 最も高い確率で予測したクラスの情報。
    print("class name : {}".format(top_1[1]))
    print("WordNet ID : {}".format(top_1[0]))
    print("Probability: {:.3f}".format(top_1[2]))

    predicted_class = np.argmax(predictions)
    cam, heatmap = grad_cam(model, x, predicted_class, "block5_conv3")
    cv2.imwrite(args.output_path, cam)
    print("GradCAM image was saved at '{}'".format(os.path.join(os.getcwd(), args.output_path)))

    sns.heatmap(heatmap, xticklabels=False, yticklabels=False, cmap="coolwarm")
    plt.savefig(args.heatmap_path)
    print("Heatmap image was saved at '{}'".format(os.path.join(os.getcwd(), args.heatmap_path)))
