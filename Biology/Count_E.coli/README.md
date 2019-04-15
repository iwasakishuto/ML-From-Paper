# E.coil's Counter
Last week, I cultured E.coil and counted them. It was very boring and hard time for me to count too small and many colonies. Tha's why I applied Machine Learning and made "E.coil's Counter."

<img src="./img/E.coil.png">

# Method
## Approach 1: threshold
If we use a global value as threshold value, it will be strongly influenced by the shooting environment, so I use <b>different thresholds for different regions of the same image.</b> It gives us better results for images with varying illumination.

### Use `adaptiveThreshold` method in OpenCV.

| params          | role                                                                                 |
|:--------------- |:------------------------------------------------------------------------------------ |
| `Adaptive Method` | `cv.ADAPTIVE_THRESH_MEAN_C` or `cv.ADAPTIVE_THRESH_GAUSSIAN_C`                       |
| `Block Size`      | It decides the size of neighbourhood area. (MUST A ODD NUMBER)                       |
| `C`               | It is just a constant which is subtracted from the mean or weighted mean calculated. |

### Results

| Block Size | Number of Colonies | image |
| ---------- | ------------------ | ----- |
| 7          | 42                 |<img src="./img/th-42-E.coil.png">       |
| 9          | 45                 |<img src="./img/th-45-E.coil.png">       |
| 11         | 57                 |<img src="./img/th-57-E.coil.png">       |
| 13         | 77                 |<img src="./img/th-77-E.coil.png">       |
| 15         | 85                 |<img src="./img/th-85-E.coil.png">       |

There is no meaning about colors.

### How to use?

```sh
$ python Count-by-threshold.py \
--input_path img/E.coil.png \
--output_path img/{}-means-{}-E.coil.png \
--Block_size 9 \
--C 2
```

## Approach 2: K-means
Apply image segmentation by K-means to distinguish the E.coil's region or not.
1. Examine E.coil's color(BGR) in the image (I used [Digital Color Meter](https://support.apple.com/guide/digital-color-meter/welcome/mac)).
2. Apply K-means segmentation to the image.
3. Change the cluster's color closest to the E.colis' to white, and the others' to black.
4. Distinguish E.coils or not.

### Results
| K   | Number of Colonies | image |
| --- | ------------------ | ----- |
| 5   | 155                |<img src="./img/5-means-155-E.coil.png">       |
| 10  | 52                 |<img src="./img/10-means-52-E.coil.png">       |
| 15  | 86                 |<img src="./img/15-means-86-E.coil.png">       |
| 20  | 75                 |<img src="./img/20-means-75-E.coil.png">       |

## How to use?

```sh
$ python Count-by-K-means.py img/E.coil.png \
--input_path img/E.coil.png \
--output_path img/{}-means-{}-E.coil.png \
--K 15 \
--RGB 222.219.203
```

## Reference
[Image Thresholding — OpenCV-Python Tutorials 1 documentation](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html)
