from mask import mask 
import matplotlib.pyplot as plt 
import scipy.misc as sci
import cv2
from time import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
Mask = mask()

inp = 'images/inputs/korean.jpg'
mask = 'images/masks/dmask2.JPG'

inp = sci.imread(inp)
mask = sci.imread(mask)
tic = time()
print('==>start<==')
print(inp.shape)
out = Mask.apply_mask(inp)

print(f'time=> {round(time() - tic, 2)}s')

#sci.imsave('out.jpg', out)

plt.imshow(out)

plt.show()     