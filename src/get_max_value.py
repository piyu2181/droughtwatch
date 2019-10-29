import numpy as np
import os

# read all bands
def maxvalue(root):
    
    dirlist = lambda di: [os.path.join(di, file) for file in os.listdir(di)]
    im_pths = dirlist(root)
    array = []
    for path in im_pths:
        im = np.load(path) 
        maxval = np.max(im)
        array.append(maxval)
        
    #print('type:',array.dtype,'max', maxValue)
    #print('shape',array.shape)
    print(array)
    max_pixel = max(array)
    return max_pixel
max_val = maxvalue('/home/debjani/Desktop/droughtwatch/data/train/images')
    