"""
Train the model
CREDIT: https://github.com/omoindrot/
"""
import os
import numpy as np #hogehoge
import argparse

import tensorflow as tf

from model.input_fn import input_fn
from model.input_fn import test_input_fn
from model.model_fn import model_fn # For 1ch images
from model.vgg_model_fn import vgg_model_fn # For 3ch images
from model.utils import Params

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='params/batch_all',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/',
                    help="Directory containing the dataset")


if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    model = model_fn if params.input_channels == 1 else vgg_model_fn

    # Define the model
    tf.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=args.model_dir,
                                    save_summary_steps=params.save_summary_steps)
    estimator = tf.estimator.Estimator(model, params=params, config=config)

    # Train the model
    tf.logging.info("Starting training for {} epoch(s).".format(params.num_epochs))
    estimator.train(lambda: input_fn(args.data_dir, params))
    
    # Evaluate the model on the test set
    tf.logging.info("Evaluation on test set.")
    res = estimator.evaluate(lambda: test_input_fn(args.data_dir, params))
    for key in res:
        print("{}: {}".format(key, res[key]))
