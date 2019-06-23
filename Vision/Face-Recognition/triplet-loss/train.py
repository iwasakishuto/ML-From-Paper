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
from model.model_fn import model_fn
from model.utils import Params


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='logger/base_model',
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

    # Define the model
    tf.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=args.model_dir,
                                    save_summary_steps=params.save_summary_steps)
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

    # Train the model
    tf.logging.info("Starting training for {} epoch(s).".format(params.num_epochs))

    estimator.train(lambda: input_fn(args.data_dir, params))
    
    # Evaluate the model on the test set
    tf.logging.info("Evaluation on test set.")
    res = estimator.evaluate(lambda: test_input_fn(args.data_dir, params))
    for key in res:
        print("{}: {}".format(key, res[key]))

    #==========================
    tf.logging.info("Predicting")
    predictions = estimator.predict(lambda: test_input_fn(args.data_dir, params))

    # TODO (@omoindrot): remove the hard-coded 10000
    embeddings = np.zeros((25, params.embedding_size))
    for i, p in enumerate(predictions):
        embeddings[i] = p['embeddings']
        print(i)
    
    np.save("hoge_vectors.npy", embeddings)