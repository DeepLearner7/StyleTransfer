# StyleTransfer
This repository is implementation of style transfer part from the paper <a href="https://arxiv.org/abs/1603.08155">Perceptual Losses for Real-Time Style Transfer and Super-Resolution</a>. The model architecture also contains <a href="https://arxiv.org/abs/1703.06868">Adaptive Instance Normalization</a> technique after each convolution layer.

Basic idea is to train a network for particular style image so to get stylized image, a single forward network is required and it can perform almost realtime. Using ADN(Adaptive Instance Normalization), resultant image quality can be increased.
Network can be trained for different style images. Some pretrained checkpoints can be found <a href="https://drive.google.com/open?id=1kjeNLTWYVnUzU2aKDsv1X-O3VV2zhcgu">here</a>.


### Result
<div align='center'>
  <img src='static/input/dp_17.jpg' height="225px"> &nbsp  &nbsp
  <img src='static/style/udnie.jpg' height="225px">&nbsp &nbsp &nbsp &nbsp &nbsp
  <img src='static/output/dp_15.jpg' height="225px">
</div>





### Prerequisites

```
flask- 0.12.2
tensorflow 1.3.0
```

### Installing

Clone this repo and make sure you have installed prerequisites packages.

## Running the tests
Run main.py file and open localhost in browser.
