from PIL import Image, ImageFilter
import numpy as np
import math


def psnr(compressed,origin):

    mse = 0

    for rownum in range(len(compressed)):
       for colnum in range(len(compressed[rownum])):
           mse += math.pow((origin[rownum][colnum] - compressed[rownum][colnum]),2)

       # print(mse)

    mse = mse/(len(origin)*len(origin[0]))

    res = 10 * math.log10(math.pow(255,2)/mse)

    return res


#Read image
im = Image.open( 'hymenoptera_data/train/ants/0013035.jpg' )
#Display image
#im.show()
#Saving the filtered image to a new file
im.save( 'compress2jpg.jpg', 'JPEG' )
cmp_im = Image.open('compress2jpg.jpg')
im = np.array(im,np.int16).transpose(2,0,1)
cmp_im = np.array(cmp_im,np.int16).transpose(2,0,1)
print("PSNR: ", psnr(im[0], cmp_im[0]))
