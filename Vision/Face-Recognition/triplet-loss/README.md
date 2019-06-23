# Triplet Loss
In face recognition domain, triplet loss is very powerful and [FaceNet](https://arxiv.org/abs/1503.03832) achieved state-of-the-art performance using only 128-bytes per face!! I wanted to try it for long, so here we go :)

### Why not softmax??
|softmax cross entropy|triplet loss|
|:--|:--|
|<b>fixed number</b> of classes|<b>variable number</b> of classes|

### Definition
$$\mathcal{L}=\max (d(a, p)-d(a, n)+\operatorname{margin}, 0)$$

- $a$ : an <font color="red"><b>anchor</b></font>
- $p$ : a <font color="red"><b>positive</b></font> of the same class as the anchor
- $n$ : a <font color="red"><b>negative</b></font> of a different class with the anchor

If you want to check whether (i,j,k) is valid triplet, run the function below.

### Triplet mining
The goal of the triplet loss is to make sure that <b>"two samples with same label are close"</b>, and <b>"two samples with different label are far away."</b> However, we <font color="red"><b>don't want to push the train embeddings of each label to collapse into very small clusters.</b></font> Therefore, we introduce "margin".

|name|relationship|brief content|
|:--|:--:|:--|
|easy triplets|$d(a,p) + \mathrm{margin} < d(a,n)$|loss is 0|
|semi-hard triplets|$d(a,p) < d(a,n) < d(a,p) + \mathrm{margin}$|p is closer to the a but have loss|
|hard triplets|$d(a,n) < d(a,p)$|n is closer to the a than p|

Each of these definitions depend on <font color="red"><b>where the negative is</b></font>, relatively to the anchor and positive, so focus on negative only.

<img src="https://omoindrot.github.io/assets/triplet_loss/triplets.png">

### Online and Offline
#### Offline
We prepare the set of triplets $(i, j, k)$ <b>at the beginning of each epoch for instance</b>, so we have to compute $3B$ embeddings to get the $B$ triplets.

```python
anchors, positives, negatives = prepare_triplets()

d_pos = tf.reduce_sum(tf.square(anchors - positives), 1)
d_neg = tf.reduce_sum(tf.square(anchors - negatives), 1)

loss = tf.maximum(0.0, margin + d_pos - d_neg)
loss = tf.reduce_mean(loss)
```

#### Online
We only prepare $B$ embeddings, and
The idea here is to compute useful triplets on the fly, for each batch of inputs. Given a batch of $B$ iamges, we compute the $B$ embeddings and we then can find a maximum of $B^3$ triplets.

## Reference
Special thanks goes to <b>Olivier Moindrot</b>. His nice blog is [here](https://omoindrot.github.io/triplet-loss).
