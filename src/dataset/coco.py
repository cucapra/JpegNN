import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import glob
import pickle
import numpy as np
import nltk
import itertools as it
from PIL import Image
import matplotlib.pyplot as plt 
from pycocotools.coco import COCO
 
path = '/data/zhijing/coco/train/'
coco = torchvision.datasets.CocoDetection(root='/data/datasets/COCO_2014/train2014/',
    annFile='/data/datasets/COCO_2014/annotations/instances_train2014.json')
    #transform=transforms.ToTensor())
print('Number of samples: ', len(coco))
for i in range(len(coco)):
    img, target = coco[i] # load ith sample
    if(len(target)==0):
        continue
        #plt.imshow(img)
        #plt.show()
    keyfunc = lambda x: x['category_id']
    groups = it.groupby(sorted(target, key=keyfunc), keyfunc)
    groups=[{'category_id':k, 'area':sum(x['area'] for x in g)} for k, g in groups]
    groups = sorted(groups, key=lambda k: k['area'], reverse=True)

    if(groups[0]['area'] < 30000 or
       len(groups)>=2 and groups[0]['area']/groups[1]['area'] < 1.8):
        continue
    #print(groups)
    #plt.imshow(img)
    #plt.show()
    #print(target, 'hello')
    save_dir = path+str(target[0]['category_id'])+'/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir )
        print("Directory ", save_dir, " Created ") 
    
    img.save(save_dir+str(target[0]['image_id'])+'.jpg')
#print("Image Size: ", img.shape)
#print(target[0]['category_id'])



