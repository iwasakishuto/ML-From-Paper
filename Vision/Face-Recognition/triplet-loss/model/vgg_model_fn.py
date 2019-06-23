"""
Define the vgg model.
CREDIT: https://github.com/omoindrot/
"""
import numpy as np
import tensorflow as tf

from model.triplet_loss import batch_all_triplet_loss
from model.triplet_loss import batch_hard_triplet_loss


def build_vgg_model(is_training, images, params):
    """
    Compute outputs of the model (embeddings for triplet loss).
    ====================================================
    @params is_training: (bool) training or not.
    @params images     : (dict) outputs of "tf.data"
    @params params     : (Params) hyperparameters
    ====================================================
    @returns out       : (tf.Tensor) output of the model
    """
    #===================================================
    # Define some functions for vgg.
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(conv, weight):
        return tf.nn.conv2d(conv, weight, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(conv):
        return tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #===================================================

    with tf.name_scope('conv1_1') as scope:
        weight = weight_variable([3, 3, 3, 64])
        bias = bias_variable([64])
        conv = conv2d(images, weight)
        out = tf.nn.bias_add(conv, bias)
        conv1_1 = tf.nn.relu(out, name=scope)

    with tf.name_scope('conv1_2') as scope:
        weight = weight_variable([3, 3, 64, 64])
        bias = bias_variable([64])
        conv = conv2d(conv1_1, weight)
        out = tf.nn.bias_add(conv, bias)
        conv1_2 = tf.nn.relu(out, name=scope)

    with tf.name_scope('pool1') as scope:
        pool1 = max_pool_2x2(conv1_2)

    with tf.name_scope('conv2_1') as scope:
        weight = weight_variable([3, 3, 64, 128])
        bias = bias_variable([128])
        conv = conv2d(pool1, weight)
        out = tf.nn.bias_add(conv, bias)
        conv2_1 = tf.nn.relu(out, name=scope)

    with tf.name_scope('conv2_2') as scope:
        weight = weight_variable([3, 3, 128, 128])
        bias = bias_variable([128])
        conv = conv2d(conv2_1, weight)
        out = tf.nn.bias_add(conv, bias)
        conv2_2 = tf.nn.relu(out, name=scope)

    with tf.name_scope('pool2') as scope:
        pool2 = max_pool_2x2(conv2_2)

    with tf.name_scope('conv3_1') as scope:
        weight = weight_variable([3, 3, 128, 256])
        bias = bias_variable([256])
        conv = conv2d(pool2, weight)
        out = tf.nn.bias_add(conv, bias)
        conv3_1 = tf.nn.relu(out, name=scope)

    with tf.name_scope('conv3_2') as scope:
        weight = weight_variable([3, 3, 256, 256])
        bias = bias_variable([256])
        conv = conv2d(conv3_1, weight)
        out = tf.nn.bias_add(conv, bias)
        conv3_2 = tf.nn.relu(out, name=scope)

    with tf.name_scope('conv3_3') as scope:
        weight = weight_variable([3, 3, 256, 256])
        bias = bias_variable([256])
        conv = conv2d(conv3_2, weight)
        out = tf.nn.bias_add(conv, bias)
        conv3_3 = tf.nn.relu(out, name=scope)

    with tf.name_scope('pool3') as scope:
        pool3 = max_pool_2x2(conv3_3)

    with tf.name_scope('conv4_1') as scope:
        weight = weight_variable([3, 3, 256, 512])
        bias = bias_variable([512])
        conv = conv2d(pool3, weight)
        out = tf.nn.bias_add(conv, bias)
        conv4_1 = tf.nn.relu(out, name=scope)

    with tf.name_scope('conv4_2') as scope:
        weight = weight_variable([3, 3, 512, 512])
        bias = bias_variable([512])
        conv = conv2d(conv4_1, weight)
        out = tf.nn.bias_add(conv, bias)
        conv4_2 = tf.nn.relu(out, name=scope)

    with tf.name_scope('conv4_3') as scope:
        weight = weight_variable([3, 3, 512, 512])
        bias = bias_variable([512])
        conv = conv2d(conv4_2, weight)
        out = tf.nn.bias_add(conv, bias)
        conv4_3 = tf.nn.relu(out, name=scope)

    with tf.name_scope('pool4') as scope:
        pool4 = max_pool_2x2(conv4_3)

    with tf.name_scope('conv5_1') as scope:
        weight = weight_variable([3, 3, 512, 512])
        bias = bias_variable([512])
        conv = conv2d(pool4, weight)
        out = tf.nn.bias_add(conv, bias)
        conv5_1 = tf.nn.relu(out, name=scope)

    with tf.name_scope('conv5_2') as scope:
        weight = weight_variable([3, 3, 512, 512])
        bias = bias_variable([512])
        conv = conv2d(conv5_1, weight)
        out = tf.nn.bias_add(conv, bias)
        conv5_2 = tf.nn.relu(out, name=scope)

    with tf.name_scope('conv5_3') as scope:
        weight = weight_variable([3, 3, 512, 512])
        bias = bias_variable([512])
        conv = conv2d(conv5_2, weight)
        out = tf.nn.bias_add(conv, bias)
        conv5_3 = tf.nn.relu(out, name=scope)

    with tf.name_scope('pool5') as scope:
        pool5 = max_pool_2x2(conv5_3)

    with tf.name_scope('fc6') as scope:
        shape = int(np.prod(pool5.get_shape()[1:]))
        weight = weight_variable([shape, 4096])
        bias = bias_variable([4096])
        pool5_flat = tf.reshape(pool5, [-1, shape])
        fc6 = tf.nn.relu(tf.nn.bias_add(tf.matmul(pool5_flat, weight), bias))
        # fc6_drop = tf.nn.dropout(fc6, params.keep_prob)

    with tf.name_scope('fc7') as scope:
        weight = weight_variable([4096, 4096])
        bias = bias_variable([4096])
        fc7 = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc6, weight), bias))
        # fc7_drop = tf.nn.dropout(fc7, params.keep_prob)

    with tf.name_scope('fc8') as scope:
        out = tf.layers.dense(fc7, params.embedding_size)
        # weight = weight_variable([4096, params.embedding_size])
        # bias = bias_variable([params.embedding_size])
        # out = tf.nn.bias_add(tf.matmul(fc7, weight), bias)

    return out


def vgg_model_fn(features, labels, mode, params):
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
    with tf.variable_scope('model'):
        embeddings = build_vgg_model(is_training, images, params)
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
