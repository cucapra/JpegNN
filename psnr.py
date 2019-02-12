from PIL import Image, ImageFilter
import numpy as np
import math
import argparse
import scipy, scipy.fftpack

parser = argparse.ArgumentParser()
parser.add_argument('-c','--comparison', help='compare psnr or not', action='store_true')
args = parser.parse_args()


def psnr(compressed,origin):

    mse = 0

    for rownum in range(len(compressed)):
       for colnum in range(len(compressed[rownum])):
           mse += math.pow((origin[rownum][colnum] - compressed[rownum][colnum]),2)

       # print(mse)

    mse = mse/(len(origin)*len(origin[0]))

    res = 10 * math.log10(math.pow(255,2)/mse)

    return res

def save(im, name):
    im = Image.fromarray(im, 'RGB')
    im.save(name)

def compressJ(im, name):
    im = Image.fromarray(im, 'RGB')
    im.save(name, 'JPEG')
    im = Image.open(name)
    return im

def dct2(a):
        return scipy.fftpack.dct( scipy.fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )
#################################
### example ####
#################################
#Read image
if args.comparison:
    my_im = Image.open('myJpeg.jpg')
    my_im = np.array(my_im,np.int16).transpose(2,0,1)
    lib_im = Image.open( 'libJpeg.jpg' )
    lib_im = np.array(lib_im, np.int32).transpose(2,0,1)
    #for i in range(lib_im.shape[1]):
    print(my_im[0])
    print(lib_im[0])
    print("===")

    org_im = Image.open( 'org.bmp' )
    org_im = np.array(org_im,np.int16).transpose(2,0,1)
   
    imsize = org_im.shape
    dct = np.zeros(imsize)
    
#    # Do 8x8 DCT on image (in-place)
#    for i in np.r_[:imsize[0]:8]:
#        for j in np.r_[:imsize[1]:8]:
#            dct[i:(i+8),j:(j+8)] = dct2( org_im[i:(i+8),j:(j+8)] ) 
#    print(dct[0])
    print("PSNR (my - lib): ", psnr(my_im[0], lib_im[0]))
    print("PSNR (org - lib):", psnr(org_im[0], lib_im[0]))
    print("PSNR (org - my): ", psnr(org_im[0], my_im[0]))
