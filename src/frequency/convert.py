from os import listdir
from os.path import isfile, join
from PIL import Image
mypath='/data/jenna/data/val/bicycle/'
savepath='../../data/pyjpeg50/'
for f in listdir(mypath):
    image_file=join(mypath, f)
    im1 = Image.open(image_file)
    image_save = join(savepath,f)
    im1.save(image_save,"JPEG", quality=50)
