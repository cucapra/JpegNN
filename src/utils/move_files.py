import os
import glob
import shutil
main_dir = '/data/zhijing/jenna_data/test1/'
dir_list = os.listdir(main_dir)
new_dir = '/data/zhijing/jenna_data/test4/'
if not os.path.exists(new_dir):
    os.makedirs(new_dir)
for dir_name in dir_list:
    sub_dir = main_dir+dir_name+'/'
    new_sub = new_dir+dir_name+'/'
    if not os.path.exists(new_sub):
        os.makedirs(new_sub)

    file_list = glob.glob(sub_dir+'*.png')
    i = 0
    for fname in file_list:
        shutil.move(fname,new_sub)
        i+=1
        if i == 100:
            break
