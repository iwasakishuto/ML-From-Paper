"""
Visualize the embedding space
"""
import os
import argparse
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from model.utils import Params
from model.input_fn import test_input_fn

from mpl_toolkits.mplot3d import proj3d
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


VALID_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def valid_image_directory(dirname):
    return dirname[0] != "." and dirname[-3:] != ".md"

def valid_image_filename(filename):
    return ('.' in filename) and (filename.rsplit('.',maxsplit=1)[1].lower() in VALID_EXTENSIONS)


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='params/batch_all',
                    help="Experiment directory containing params.json and model weight.")
parser.add_argument('--data_dir', default='data/',
                    help="Directory containing the dataset")
parser.add_argument('--save_filename', default='embedding-space{}'.format(datetime.now().isoformat(timespec='seconds')),
                    help="The file name of plot images.")
parser.add_argument('--color_seed', default=111, help="The random seed for plot colors.")
parser.add_argument('--title', default="Embedding space", help="Title for embedding space.")



if __name__ == '__main__':
    #=== Initialization ===
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    #=== Load the parameters ===
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    #=== Check whether there is directory === 
    assert os.path.isdir("visualize"), "Please make {}.".format(os.path.join(os.getcwd(), "visualize"))
    filename = os.path.join("visualize", args.save_filename) + ".png"

    #=== Define the model ===
    tf.logging.info("Loading the model structure...")
    if params.model == "resnet_v2":
        from model.model_fn_resnet_v2 import resnet_v2_model_fn as model
    elif params.model == "resnet_v1":
        from model.model_fn_resnet_v1 import resnet_v1_model_fn as model
    elif params.model == "vgg16":
        from model.model_fn_vgg16 import vgg16_model_fn as model
    elif params.model == "mono":
        from model.model_fn_mono import mono_model_fn as model
    elif params.model == "inception_v3":
        from model.model_fn_inception_v3 import inception_v3_model_fn as model
    else:
        tf.logging.info("Your model name {} couldn't understand.".format(params.model))

    tf.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=args.model_dir,
                                    save_summary_steps=params.save_summary_steps)
    estimator = tf.estimator.Estimator(model, params=params, config=config)

    #=== Prediction and calculate embeddings ===
    tf.logging.info("Predicting...")
    predictions = estimator.predict(lambda: test_input_fn(args.data_dir, params))
    embeddings = [p['embeddings'] for p in predictions]
    tf.logging.info("data num: {}".format(len(embeddings)))
    tf.logging.info("embedding shape: {}".format(embeddings[0].shape))

    #=== meta data for images ===
    label_names = [par for par in os.listdir(args.data_dir) if valid_image_directory(par)]
    all_image_labels = [label for label in label_names for filename in os.listdir(os.path.join(args.data_dir, label)) if valid_image_filename(filename)]

    #=== Transform embeddings to 3-D vectors ===
    tf.logging.info("Transforming embeddings to 3-D vectors...")
    pca = PCA(n_components=3)
    pca.fit(embeddings)
    Xd = pca.transform(embeddings)

    #=== Visualize embeddings ===
    tf.logging.info("Plotting...")
    fig = plt.figure(num=None, figsize=(15, 15), dpi=150, facecolor='w', edgecolor='k')
    ax = fig.gca(projection='3d')

    color_seed = args.color_seed
    np.random.seed(color_seed)
    color = [0,0,0]
    name = ""

    for label, X in zip(all_image_labels, Xd):
        if name != label:
            name = label
            color = np.random.randint(0,255,3)/255
            ax.text(X[0], X[1], X[2], name, color=color, fontsize=20)
        ax.scatter([X[0]], [X[1]], [X[2]], marker="o", linestyle='None', color=color, s=100)

    plt.title(args.title, fontsize=30)
    plt.savefig(filename)
    tf.logging.info("Completed.")
