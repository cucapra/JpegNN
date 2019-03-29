import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser(description = \
                'Test JpegLayer Compression Ratio')
parser.add_argument('--data_dir', '-d', type=str,\
    default='/data/jenna/data/val/', \
    help='Directory of the input data. \
    String. Default: /data/jenna/data/val')
parser.add_argument('--out', '-o', type=str,\
    default='jpeg50', \
    help='Name of histgram. \
    String. Default: jpeg50')

args,unparsed = parser.parse_known_args()
def histgram(path):
    comp = glob.glob(path)
    print(comp)
    x = []
    for name in comp:
        print(name)
        x.append(os.path.getsize(name))
    x=np.array(x)
    num_bins = 10
    plt.hist(x, num_bins)
    plt.xlabel('File size in Byte with mean '+str(int(x.mean())))
    plt.ylabel('# of files')
    plt.savefig('histgrams/'+args.out+'.png')
    plt.show()
#histgram("/data/zhijing/compression/uncomp_coco/*.bmp")
#histgram("/data/zhijing/compression/jpegmy2_coco/*.jpg")
#histgram("/data/zhijing/compression/jpeg20_coco/*.jpg")
histgram(args.data_dir+"/*.jpg")
#histgram("../../data/jpeg50/*.jpg")
