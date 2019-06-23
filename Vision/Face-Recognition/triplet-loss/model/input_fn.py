"""
Create the input data pipeline using "tf.data"
"""

import os
import tensorflow as tf

VALID_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def valid_image_directory(dirname):
    return dirname[0] != "." and dirname[-3:] != ".md"

def valid_image_filename(filename):
    return ('.' in filename) and (filename.rsplit('.',maxsplit=1)[1].lower() in VALID_EXTENSIONS)


def input_fn(data_dir, params):
    label_names = [par for par in os.listdir(data_dir) if valid_image_directory(par)]
    label2id = dict((name, index) for index,name in enumerate(label_names))
    
    all_image_paths = [os.path.join(data_dir, label, filename) for label in label_names for filename in os.listdir(os.path.join(data_dir, label)) if valid_image_filename(filename)]
    all_image_labels = [label2id[label] for label in label_names for filename in os.listdir(os.path.join(data_dir, label)) if valid_image_filename(filename)]
    
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int32))
    
    dataset = tf.data.Dataset.zip((image_ds, label_ds))
    
    dataset = dataset.shuffle(params.train_size)  # whole dataset into the buffer
    dataset = dataset.repeat(params.num_epochs)  # repeat for multiple epochs
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    return dataset

def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, [28, 28])
    image /= 255.0  # normalize to [0,1] range

    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

def test_input_fn(data_dir, params):
    label_names = [par for par in os.listdir(data_dir) if valid_image_directory(par)]
    label2id = dict((name, index) for index,name in enumerate(label_names))
    
    all_image_paths = [os.path.join(data_dir, label, filename) for label in label_names for filename in os.listdir(os.path.join(data_dir, label)) if valid_image_filename(filename)]
    all_image_labels = [label2id[label] for label in label_names for filename in os.listdir(os.path.join(data_dir, label)) if valid_image_filename(filename)]
    
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int32))
    
    dataset = tf.data.Dataset.zip((image_ds, label_ds))
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    return dataset
