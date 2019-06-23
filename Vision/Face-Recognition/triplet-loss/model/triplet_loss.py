"""
Define functions to create the triplet loss with online triplet mining.
CREDIT: https://github.com/omoindrot/
"""

import tensorflow as tf

# Return a 2D mask where mask[a, p] is True if p is valid positive.
def _get_anchor_positive_triplet_mask(labels):
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    
    a_equal_p = tf.expand_dims(labels, 0) # shape = (1, batch_size)
    p_equal_a = tf.expand_dims(labels, 1) # shape = (batch_size, 1)

    # Check if labels[i] == labels[j]
    labels_equal = tf.equal(a_equal_p, p_equal_a) # Using broadcast

    # Combine the two masks
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask


# Return a 2D mask where mask[a, n] is True if n is valid negative.
# If label is not equal, two sample must be distinct.
def _get_anchor_negative_triplet_mask(labels):
    a_equal_n = tf.expand_dims(labels, 0) # shape = (1, batch_size)
    n_equal_a = tf.expand_dims(labels, 1) # shape = (batch_size, 1)
    
    # Check if labels[i] != labels[k]
    labels_equal = tf.equal(a_equal_n, n_equal_a) # Use broadcast

    mask = tf.logical_not(labels_equal)

    return mask


# Return a 3D mask where mask[a, p, n] is True if the triplet (a, p, n) is valid.
def _get_triplet_mask(labels):
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2) # shape = (batch_size, batch_size, 1)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1) # shape = (batch_size, 1, batch_size)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0) # shape = (1, batch_size, batch_size)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask


# Triplet loss depends on the distance,
# so first prepare the function which could compute the pairwise distance matrix efficiently
def _pairwise_distances(embeddings, squared=False):
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings)) # dot product between all embeddings
    square_norm = tf.diag_part(dot_product) # extract only diagonal elements so that get respective norms.
    
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)
    distances = tf.maximum(distances, 0.0) # for computational errors

    if not squared:
        mask = tf.to_float(tf.equal(distances, 0.0)) # for avoid the gradient of sqrt is infinite, 
        distances = distances + mask * 1e-16 # add a small epsilon where distances == 0.0
        distances = tf.sqrt(distances)
        distances = distances * (1.0 - mask) # Correct the epsilon added

    return distances


# Compute the all valid triplet loss and average them.
def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    pairwise_dist = _pairwise_distances(embeddings, squared=squared) # Get the pairwise distance matrix

    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2) # shape = (batch_size, batch_size, 1)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1) # shape = (batch_size, 1, batch_size)

    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin # Using broadcast.

    # Remove(Put zero) the invalid triplets.
    mask = _get_triplet_mask(labels)
    mask = tf.to_float(mask)
    triplet_loss = tf.multiply(mask, triplet_loss)

    # Remove negative losses (which are the easy triplets).
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count the number of positive triplets.
    valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16)) # val > 0 means positive triplets
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    rate_of_positive = num_positive_triplets / (num_valid_triplets + 1e-16) # Rate of positive triplets from valid.

    ave_triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return ave_triplet_loss, rate_of_positive


# Compute only the triplets (hardest positive, hardest negative, anchor) for each anchor.
def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):
    pairwise_dist = _pairwise_distances(embeddings, squared=squared) # Get the pairwise distance matrix

    #=== hardest positive ===
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels) # mask(same label and distinct)
    mask_anchor_positive = tf.to_float(mask_anchor_positive)
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist) # Apply 0 to invalid element, and 1 to valid one
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True) # shape = (batch_size, 1)
    print("hardest_positive_dist.shape: {}".format(hardest_positive_dist.shape))

    #=== hardest negative ===
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels) # mask(different label and distinct)
    mask_anchor_negative = tf.to_float(mask_anchor_negative)
    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True) # Find the maximum value in each row.
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative) # Add max to invalid. 
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True) # shape = (batch_size, )
    print("hardest_negative_dist.shape: {}".format(hardest_negative_dist.shape))

    #=== Combine ===
    triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0) # Find biggest d(a, p) and smallest d(a, n)
    triplet_loss = tf.reduce_mean(triplet_loss)

    return triplet_loss
