import re
import numpy as np
from ast import literal_eval

f = open("qtable_fail.txt", "r")
#a=f.read()
a = re.sub('\s+', ', ', f.read())
print(a)
print(literal_eval(a))
a = np.array(literal_eval(a))
print(a*255)
