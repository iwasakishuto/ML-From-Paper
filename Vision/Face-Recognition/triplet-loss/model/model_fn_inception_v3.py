"""Use tensorflow Hub to implement Inception Resnet V3 architecture."""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from model.triplet_loss import batch_all_triplet_loss
from model.triplet_loss import batch_hard_triplet_loss

def inception_v3_model_fn(features, labels, mode, params):
    """
    Model function for tf.estimator
    ====================================================
    @params features   : (Tensor) input batch of images
    @params labels     : (Tensor) labels of the input images
    @params mode       : (tf.estimator.ModeKeys) mode of {TRAIN, EVAL, PREDICT}
    @params params     : (Params) hyperparameters
    ====================================================
    @returns model_spec: (tf.estimator.EstimatorSpec)
    """

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # Reshape and arrange the images.
    images = features
    images = tf.reshape(images, [-1, params.image_size, params.image_size, 3])
    assert images.shape[1:] == [params.image_size, params.image_size, 3], "{}".format(images.shape)

    # define the layers of the model
    module = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/3", trainable=True, tags={"train"})
    
    with tf.variable_scope('model'):
        outputs = module(inputs=dict(images=images, batch_norm_momentum=0.997), signature="image_feature_vector_with_bn_hparams")
        embeddings = tf.layers.dense(outputs, params.embedding_size)
        
    embedding_mean_norm = tf.reduce_mean(tf.norm(embeddings, axis=1))
    tf.compat.v1.summary.scalar("embedding_mean_norm", embedding_mean_norm)

    # ==================================================
    # [PREDICT MODE] Return the embeddings.
    if mode == tf.estimator.ModeKeys.PREDICT: 
        predictions = {'embeddings': embeddings}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    labels = tf.cast(labels, tf.int64)

    # Define triplet loss
    if params.triplet_strategy == "batch_all":
        loss, fraction = batch_all_triplet_loss(labels, embeddings, margin=params.margin, squared=params.squared)
    elif params.triplet_strategy == "batch_hard":
        loss = batch_hard_triplet_loss(labels, embeddings, margin=params.margin, squared=params.squared)
    else:
        raise ValueError("Triplet strategy not recognized: {}".format(params.triplet_strategy))

    # Metrics for evaluation using tf.metrics (average over whole dataset)
    # TODO: some other metrics like rank-1 accuracy?
    with tf.variable_scope("metrics"):
        eval_metric_ops = {"embedding_mean_norm": tf.compat.v1.metrics.mean(embedding_mean_norm)}

        if params.triplet_strategy == "batch_all":
            eval_metric_ops['fraction_positive_triplets'] = tf.compat.v1.metrics.mean(fraction)
    
    # ==================================================
    # [EVAL MODE] Return the evaluation metrics.
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

    # ==================================================
    # [TRAIN MODE] Return the evaluation metrics.
    # Summaries for training
    tf.compat.v1.summary.scalar('loss', loss)
    if params.triplet_strategy == "batch_all":
        tf.compat.v1.summary.scalar('fraction_positive_triplets', fraction)

    tf.compat.v1.summary.image('train_image', images, max_outputs=1)

    # Define training step that minimizes the loss with the Adam optimizer
    optimizer = tf.compat.v1.train.AdamOptimizer(params.learning_rate)
    global_step = tf.compat.v1.train.get_global_step()
    if params.use_batch_norm:
        # Add a dependency to update the moving mean and variance for batch normalization
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(loss, global_step=global_step)
    else:
        train_op = optimizer.minimize(loss, global_step=global_step)

    model_spec = tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    return model_spec
