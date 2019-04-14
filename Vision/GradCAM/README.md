# GradCAM
#### \~ Gradient-weighted Class Activation Mapping \~

## Papers
- [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)

## Overview
1. 「予測への影響が大きいピクセル」をヒートマップを利用して可視化したい。
2. 「ピクセル値に微小変化を加えたときに予測に生じる変化の大きさ」が、「予測への影響の大きさ」を表せるのでは？
3. 「CNNの層を増すごとに複雑な特徴量を捉えることができると考えられている」 & 「全結合層では空間的な情報が失われる」ことより、「最後の畳み込み層の出力結果(特徴量マップ)」を利用するのが良いのでは？？
><b>A number of previous works have asserted that deeper representations in a CNN capture higher-level visual constructure.</b> Furthermore, <b>convolutional features naturally retain spatial information which is lost in fully-connected layers</b>, so we can expect <b>the last convolutional layers to have the best compromise between high-level semantics and detailed spatial information.</b>
4. 特徴量マップの各値において微小変化を加えた時のクラス $c$ に対する予測結果への影響の大きさ(微分係数 $\alpha_k^c$)を平均化することで、その特徴マップの重要度が計算できる。

<div style="text-align: center;"><a href="https://www.codecogs.com/eqnedit.php?latex=$$&space;\alpha_{k}^{c}=\frac{1}{Z}&space;\sum_{i}&space;\sum_{j}&space;\quad&space;\frac{\partial&space;y^{c}}{\partial&space;A_{i&space;j}^{k}}&space;$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$&space;\alpha_{k}^{c}=\frac{1}{Z}&space;\sum_{i}&space;\sum_{j}&space;\quad&space;\frac{\partial&space;y^{c}}{\partial&space;A_{i&space;j}^{k}}&space;$$" title="$$ \alpha_{k}^{c}=\frac{1}{Z} \sum_{i} \sum_{j} \quad \frac{\partial y^{c}}{\partial A_{i j}^{k}} $$" /></a></div>

5. これに、特徴マップをかけ、ReLUを通す(これによって、重要なところだけしか残さない)ことで、ヒートマップを作ることが可能。

<div style="text-align: center;"><a href="https://www.codecogs.com/eqnedit.php?latex=$$&space;L_{\mathrm{Grad}&space;\mathrm{CAM}}^{c}=\operatorname{ReLU}\left(\sum_{k}&space;\alpha_{k}^{c}&space;A^{k}\right)&space;$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$&space;L_{\mathrm{Grad}&space;\mathrm{CAM}}^{c}=\operatorname{ReLU}\left(\sum_{k}&space;\alpha_{k}^{c}&space;A^{k}\right)&space;$$" title="$$ L_{\mathrm{Grad} \mathrm{CAM}}^{c}=\operatorname{ReLU}\left(\sum_{k} \alpha_{k}^{c} A^{k}\right) $$" /></a></div>
