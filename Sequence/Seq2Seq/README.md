# seq2seq
<b>\~Sequence to Sequence\~</b>
Deep Neural Networks are

## Papers
- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)

## Overview
A Sequence to Sequence network is a model consisting of two separate RNNs called the <b>encoder</b> and <b>decoder.</b> The encoder encodes the input sequence, and the decoder produces the target sequence.
> As an example, let's translate `how are you` (English) to `Comment vas tu` (French).

### Encoder
Each word from the input sequence is thought <b>as a one-hot vector</b> (That's why we need a vocabulary), so in this case, we have 3 words, thus our input will be transformed into [w0, w1, w2]. Then, we simply run an LSTM and store the last hidden state (e2), this will be our encoder representation e. (e = e2)
<div align="center"><img src="./img/vanilla-encoder.svg" width=70%></div>

### Decoder
Now that we have a <b>vector e that captures the meaning of the input sequence</b>, we'll use it to generate the target sequence word by word. Feed to another LSTM cell: e as hidden state and a special vector w_\<sos\> are inputs. The LSTM computes the next hidden state h0. Then, we apply some function g so that s0 size is equal to the vocabulary. After that, we apply a softmax to s0 to normalize it into avector of probabilities p0. Get a vector w_i0 = w_<comment> and repeat the procedure. <b>The decoding stops when the predicted word is a special vector w_\<eos\></b>

<div align="center"><img src=http://latex2png.com/output//latex_4e209d0847608208736060e21e9c2422.png width=30%></div>
<div align="center"><img src="./img/vanilla-decoder.svg" width=70%></div>

With the seq2seq model, by <b>encoding many inputs into one vector</b>, and <b>decoding from one vector into many outputs</b>, we are freed from the constraints of sequence order and length. The encoded sequence is represented by a single vector, a single point in some N dimensional space of sequences. In an ideal case, <b>this point can be considered the "meaning" of the sequence.</b>

## Reference
- [Seq2Seq with Attention and Beam Search](https://guillaumegenthial.github.io/sequence-to-sequence.html)
- da
