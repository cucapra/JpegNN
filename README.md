A naive framework of Jpeg compression layer added to some nenural networks.

- Todo 
unit test: 
1. single image output
2. psnr to each step: rgb2ycbcr, dct, quantize

backward propagation:
1. should not depends on decompression(deepcopy?)

optimization (far in the future):
1. the code is currently ugly, make it beautiful(some day)
2. speed up
