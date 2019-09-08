import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

src1 = np.array(Image.open(filename))

noise = np.rint(np.random.normal(0, 56, src1.shape))
print(src1.shape)
mask = np.zeros(src1.shape, dtype=int)

for face in facearrays:
    mask[face[0]:face[2],face[3]:face[1]] = noise[face[0]:face[2],face[3]:face[1]]

final = np.clip(np.absolute(np.add(mask, src1)).astype(int), 0, 255)
im = Image.fromarray(final.astype('uint8'), 'RGB')
im.save("final.jpg")




