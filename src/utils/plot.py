import argparse
import glob
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dir', '--names-list', nargs='+',
default = ['quality_noft', 'train_quality','test_quality'])
#default = ['jpeg_noft'])
args = parser.parse_args()

plt.title('Quality vs Accuarcy')
color=['r','g','b']
for i,name in enumerate(args.dir):
    folder = '../../result/'+name+'/'
    result = []
    for f in glob.glob(folder+"*"):
        fd = open(f,'r')
        acc=float(re.search("Best val Acc: (.*)",fd.read()).groups()[0])
        result.append(acc*100)
    x = np.arange(len(result))*10
    plt.plot(x, result, color=color[i], label=name)
#    for j, y in enumerate(result):
#        plt.annotate(round(y*10)/10,(x[j],y), fontsize=10,color=color[i])
plt.legend()
plt.xlabel('quality')
#plt.xlabel('trial')
plt.ylabel('Acc')
plt.savefig('jpeg.png')
plt.show()
