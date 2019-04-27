# -*- coding: utf-8 -*-

from main import generate_model
from PIL import Image
from keras_preprocessing.image import img_to_array
import keras.backend as K
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_img(path, grayscale=False, target_size=None, keep_aspect_ratio=False, cval=255):
    """
    関数の概要：PIL の形式で画像を読み込む。
    @param path             ：画像へのパス
    @param grayscale        ：白黒の画像に変換する = True
    @param target_size      ：(height, width)で指定する画像サイズ。指定しなければそのまま
    @param keep_aspect_ratio：リサイズした時に元画像と同じ比率を保つか。同じならば、画像を中央に置き、cval で padding.
    @param cval             ：padding する時のピクセル値。[0,255]
    """
    img = Image.open(path)

    if grayscale:
        img=img.convert('L')
    else:
        img=img.convert('RGB') # (元画像が白黒でも、αチャンネルを含んでいても 3ch に変換。)

    if target_size:
        size=(target_size[1], target_size[0])

        if not keep_aspect_ratio:
            img = img.resize(size) # 何も考えずに変換。
        else:
            if img.width > img.height:
                if img.width < w:
                    size = (img.width, img.width)
            else:
                if img.height < h:
                    size = (img.height, img.height)

            img.thumbnail(size, Image.ANTIALIAS)
            bcg=Image.new(('L' if grayscale else 'RGB'), size, (cval if grayscale else (cval, cval, cval)))
            bcg.paste(img, ((size[0] - img.size[0])//2,
                            (size[1] - img.size[1])//2))

            if bcg.width < target_size[1]:
                bcg = bcg.resize((target_size[1], target_size[0]))
            return bcg
    return img

def grad_cam(model, img1, img2, category_index, layer_name="share_conv_2"):
    nb_classes = 2
    conv_output =  [l for l in model.layers if l.name is layer_name][0].output

    one_hot = K.one_hot([category_index], nb_classes)
    signal = tf.multiply(model.output, one_hot)
    loss = K.mean(signal)

    grads = tf.gradients(loss, conv_output)[0]
    norm_grads = tf.divide(grads, K.sqrt(K.mean(K.square(grads))) + tf.constant(1e-5))

    gradient_function = K.function([model.layers[0].input], [conv_output, norm_grads]) # img1のみを考える。

    output, grads_val = gradient_function([img1])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[:2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i] # 各特徴マップを、その重要度に比率した重みを掛けて足し合わせる。

    cam = cv2.resize(cam, (160, 60)) # 元々の入力画像のサイズに整形。
    # cam = np.maximum(cam, 0) # 0以下の値を0にする。
    # heatmap = cam / np.max(cam) # 最大値で割ることで正規化。
    heatmap = (cam - np.min(cam)) / (np.max(cam) - np.min(cam)) # heatmap を 0-1正規化する。

    # 前処理された画像を、元のRGBの画像に戻す。
    img1 = img1[0, :]
    img1 -= np.min(img1)
    img1 = np.minimum(img1, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(img1)
    cam = 255 * (cam / np.max(cam))

    return np.uint8(cam), heatmap

if __name__ == "__main__":
    model = generate_model(print_summary=False)
    model.load_weights('./weight/checkpoint_06-0.194.weight')

    path1 = "./img/false-diff-369_1573.png"
    path2 = "./img/false-diff-369_1585.png"

    img1 = load_img(path1, target_size=(160, 60), keep_aspect_ratio=False)
    img2 = load_img(path2, target_size=(160, 60), keep_aspect_ratio=False)

    img1 = img_to_array(img1) / 255
    img2 = img_to_array(img2) / 255

    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)

    img1 = img1.astype('float32')
    img2 = img2.astype('float32')

    predictions = model.predict([img1, img2])[0]
    print("The probability of same: {}".format(predictions[0]))
    predicted_class = np.argmax(predictions)

    cam, heatmap = grad_cam(model, img1, img2, predicted_class, "share_conv_2")
    cv2.imwrite("cam.png", cam)

    sns.heatmap(heatmap, xticklabels=False, yticklabels=False, cmap="coolwarm")
    plt.savefig("heatmap.png")
