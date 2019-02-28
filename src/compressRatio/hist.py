import numpy as np
import os
import glob
import matplotlib.pyplot as plt
def histgram(path):
    comp = glob.glob(path)
    x = []
    for name in comp:
        print(name)
        x.append(os.path.getsize(name))
    x=np.array(x)
    num_bins = 10
    plt.hist(x, num_bins)
    plt.xlabel('File size in Byte with mean'+str(int(x.mean())))
    plt.ylabel('# of files')
    plt.show()

histgram("../../data/jpeg20/*.jpg")
histgram("../../data/jpeg50/*.jpg")
