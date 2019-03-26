from shutil import copyfile
import os,glob,sys,getopt,json
import numpy as np
from pycocotools.coco import COCO
 
src_path = '/data/zhijing/coco/'
dst_path = '/data/zhijing/coco/actual/'

train = 'train/'
val = 'val/'

annFile='/data/datasets/COCO_2014/annotations/instances_train2014.json'
coco=COCO(annFile)

cat_list = []
for cat_id in os.listdir(src_path+train):
    img_list = [] # img_id, area for certain catogery
    for img_id in os.listdir(src_path+train+cat_id+'/'):
        img_id_int = int(img_id[:-4])
        ann_ids = coco.getAnnIds(imgIds = img_id_int)
        target = coco.loadAnns(ann_ids)
        area = 0
        for x in target:
            if x['category_id'] == int(cat_id):
                area += x['area']
        img_list.append({'img_id':img_id, 'area': area})
    img_list = sorted(img_list, key=lambda k: k['area'], reverse=True)
    if len(img_list) > 300:
        src_dir = src_path+train+cat_id+'/'
        dst_dir = dst_path+train+cat_id+'/'
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir )
            print("Directory ", dst_dir, " Created ") 
        for kv_id in range(300):
            img_id=img_list[kv_id]['img_id']
            copyfile(src_dir+img_id, dst_dir+img_id)
        cat_list.append(cat_id)

annFile='/data/datasets/COCO_2014/annotations/instances_val2014.json'
coco=COCO(annFile)
for cat_id in cat_list:
    img_list = []
    for img_id in os.listdir(src_path+val+cat_id+'/'):
        img_id_int = int(img_id[:-4])
        ann_ids = coco.getAnnIds(imgIds = img_id_int)
        target = coco.loadAnns(ann_ids)
        area = 0
        for x in target:
            if x['category_id'] == int(cat_id):
                area += x['area']
        img_list.append({'img_id':img_id, 'area': area})
    img_list = sorted(img_list, key=lambda k: k['area'], reverse=True)
    if len(img_list) > 50:
        src_dir = src_path+val+cat_id+'/'
        dst_dir = dst_path+val+cat_id+'/'
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir )
            print("Directory ", dst_dir, " Created ") 
        for kv_id in range(50):
            img_id=img_list[kv_id]['img_id']
            copyfile(src_dir+img_id, dst_dir+img_id)
    else:
        print('Exception!!!!! ', cat_id)
    
