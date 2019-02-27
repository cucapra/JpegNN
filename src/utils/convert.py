from PIL import Image
file_in = "org.bmp"
img = Image.open(file_in)
file_out = "org.png"
img.save(file_out)
