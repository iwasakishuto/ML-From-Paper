# coding: utf-8
# --------------------------------------------------------
# Triplet Loss Function
# Define functions to create the triplet loss with online triplet mining.
# --------------------------------------------------------

import tensorflow as tf

def _pairwise_distances(embeddings, squared=False):
    """
    Compute the 2D matrix of distances between all the embeddings.
    ===============================================================
    @param  embeddings： embeddings for each face. (batch_size, embed_dim)
    @param  squared   ： True : output is the pairwise squared euclidean distance matrix.
                      　 False: output is the pairwise euclidean distance matrix.
    @return distances ： distance matrix in embedding space. (batch_size, batch_size)
    """
    # Get the inner product between all embeddings. Aij = ai*aj
    inner_product = tf.matmul(embeddings, tf.transpose(embeddings))
    # Get squared L2 norm for each embedding. Bi = ai*ai
    square_norm = tf.diag_part(inner_product)
    # Compute the pairwise distance matrix. ||ai - aj||^2 = ||ai||^2  - 2 <ai, aj> + ||aj||^2
    distances = tf.expand_dims(square_norm, 1) - 2.0 * inner_product + tf.expand_dims(square_norm, 0)
    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # we need to add a small epsilon where distances == 0.0 in order to avoid inf error.
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = tf.sqrt(distances + mask * 1e-16)
        # Correct the epsilon added.
        distances = distances * (1.0 - mask)

    return distances

def _get_anchor_positive_triplet_mask(labels):
    """
    Return a 2D mask about whether they are (anchor, positive) or not.
    ===============================================================
    @param labels： labels about who? [batch_size]
    @return mask ： mask[a, p] == True : a and p are distinct and have same label.
                 　 mask[a, p] == False: a and p are not distinct or have different label.
    """
    # Check that i and j are distinct.
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # Combine the two masks
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask

def _get_anchor_negative_triplet_mask(labels):
    """
    Return a 2D mask about whether they are (anchor, negative) or not.
    If labels[i] != labels[j], i and j are definitly distinct, so we don't have to think about it.
    ===============================================================
    @param labels： labels about who? [batch_size]
    @return mask ： mask[a, p] == True : a and p are not distinct or have different label.
                 　 mask[a, p] == False: a and p are distinct and have same label.
    """
    # Check if labels[i] != labels[j]
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask

def _get_triplet_mask(labels):
    """
    Return a 3D mask whether they are (anchor, positive, negative) or not.
    ===============================================================
    @param labels： labels about who? [batch_size]
    @return mask ： mask[i, j, k] == True : i,j,k are distinct and labels[i] == labels[j], labels[i] != labels[k].
                 　 mask[i, j, k] == False: i,j,k are invalid triplet.
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)
    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)
    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask

def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    """
    Build the triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive ones.
    ===============================================================
    @param  labels       ： labels about who? [batch_size]
    @param  embeddings   ： embeddings for each face. (batch_size, embed_dim)
    @param  margin       ： margin for triplet loss
    @param  squared      ： True : output is the pairwise squared euclidean distance matrix.
                      　    False: output is the pairwise euclidean distance matrix.
    @return triplet_loss ： scalar tensor containing the triplet loss
    @return fraction_positive_triplets
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    # Compute a triplet_loss[i, j, k] (anchor=i, positive=j, negative=k)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    mask = _get_triplet_mask(labels)
    mask = tf.to_float(mask)
    triplet_loss = tf.multiply(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)

    # Rate about "N_positive_triplets / N_valid_triplets"
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)
    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets

def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):
    """
    Build the triplet loss over a batch of embeddings.
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    ===============================================================
    @param  labels       ： labels about who? [batch_size]
    @param  embeddings   ： embeddings for each face. (batch_size, embed_dim)
    @param  margin       ： margin for triplet loss
    @param  squared      ： True : output is the pairwise squared euclidean distance matrix.
                      　    False: output is the pairwise euclidean distance matrix.
    @return triplet_loss ： scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    mask_anchor_positive = tf.to_float(_get_anchor_positive_triplet_mask(labels))
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_positive_dist", tf.reduce_mean(hardest_positive_dist))

    # For each anchor, get the hardest negative
    mask_anchor_negative = tf.to_float(_get_anchor_negative_triplet_mask(labels))
    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)

    # Get final mean triplet loss
    triplet_loss = tf.reduce_mean(triplet_loss)

    return triplet_loss
