"""
Define the model.
CREDIT: https://github.com/omoindrot/
"""
import tensorflow as tf
import keras

from model.triplet_loss import batch_all_triplet_loss
from model.triplet_loss import batch_hard_triplet_loss


def build_model(is_training, images, params):
    """
    Compute outputs of the model (embeddings for triplet loss).
    ====================================================
    @params is_training: (bool) training or not.
    @params images     : (dict) outputs of "tf.data"
    @params params     : (Params) hyperparameters
    ====================================================
    @returns out       : (tf.Tensor) output of the model
    """
    out = images # shape = (?,28,28,1)
    # Define the number of channels of each convolution
    # For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool
    num_channels = params.num_channels
    bn_momentum = params.bn_momentum
    channels = [num_channels, num_channels * 2]
    for i, c in enumerate(channels):
        with tf.variable_scope('block_{}'.format(i+1)):
            out = tf.layers.conv2d(out, c, 3, padding='same') # shape = (?,28,28,32), (?, 14, 14, 64)
            if params.use_batch_norm:
                out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            out = tf.nn.relu(out) # shape = (?,28,28,32), (?, 14, 14, 64)
            out = tf.layers.max_pooling2d(out, 2, 2) # shape= (?, 14, 14, 32), (?, 7, 7, 64)
    size = params.image_size//2//2
    assert out.shape[1:] == [size, size, num_channels * 2] 

    out = tf.reshape(out, [-1, size * size * num_channels * 2])
    with tf.variable_scope('fc_1'):
        out = tf.layers.dense(out, params.embedding_size)

    return out
    # model = keras.models.Model(images, out)
    # return model


def model_fn(features, labels, mode, params):
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
    images = tf.reshape(images, [-1, params.image_size, params.image_size, 1])
    assert images.shape[1:] == [params.image_size, params.image_size, 1], "{}".format(images.shape)

    # define the layers of the model
    with tf.variable_scope('model'):
        embeddings = build_model(is_training, images, params)
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
