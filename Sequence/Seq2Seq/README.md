# seq2seq
<b>\~Sequence to Sequence\~</b>
Deep Neural Networks are

## Papers
- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)

## Overview
A Sequence to Sequence network is a model consisting of two separate RNNs called the <b>encoder</b> and <b>decoder.</b>

| Encoder | Decoder |
| ------- | ------- |
|Reads an input sequence one item at a time, and outputs a vector at each step. The final output of the encoder is kept as the context vector.|Uses this context vector (from encoder) to produce a sequence of outputs one step at a time.|
| <img src="https://cdn-images-1.medium.com/max/1600/1*3pH2NH_8i7QMxpV0TFOdxw.jpeg">        |  <img src="https://cdn-images-1.medium.com/max/1600/1*sDlV9_-PXBlt8jol-7Xjhg.jpeg">       |

When using a single RNN, there is a <b>one-to-one relationship between inputs and outputs.</b> We would quickly run into problems with different sequence orders and lengths that are common during translation.

With the seq2seq model, by <b>encoding many inputs into one vector</b>, and <b>decoding from one vector into many outputs</b>, we are freed from the constraints of sequence order and length. The encoded sequence is represented by a single vector, a single point in some N dimensional space of sequences. In an ideal case, <b>this point can be considered the "meaning" of the sequence.</b>
