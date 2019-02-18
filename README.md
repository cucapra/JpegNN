# JpegNN
A naive framework of Jpeg compression layer added to some nenural networks.

## Dependency
Pytorch, Numpy, Argparse, PIL, Matplotlib

## Usage
cnn.py 
- A normal neural network framework. One can choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]. (Actually inception is not finished because I haven't add padding yet.)
- One can choose to add jpeg layer by setting the add\_jpeg\_layer flag

jpeg\_layer.py
- The layer of jpeg implementation. 
- It contains rgb2ycbcr, subsampling(2x2 box filter), dct2, quantization(qtable as parameter); quantization back(not parameter), idct2, upsampling(2x2 duplicating), ycbcr2rgb

psnr.py
- For some simple testing on psnr between libjpeg and feature map after jpeg\_layer.

gradient.py
- Because torch.clamp() and torch.round() give 0 as gradient, this file is to customize our own clamp and round function

## Todo 
Change loss function:
- Modifying loss function so that we can keep qtable magnitude as small as possible.
- scaling factor: a*loss + b*1/magnitude
- regularization tech: look it up

Test on uncompress data:
- waiting for datasets

Jpeg verification:
- look into libjpeg at https://github.com/LuaDist/libjpeg. Currently the compressions are very close, but still not the same. 


unit test for backward propagation:
- should not depends on decompression(not sure how to test that yet)

optimization (far in the future):
- the code is currently ugly, make it beautiful(some day)
- speed up
