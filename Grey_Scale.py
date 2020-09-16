import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


img = mpimg.imread('colorbalanced.png')
gray = rgb2gray(img)
plt.imshow(gray, cmap=plt.get_cmap('gray'))

plt.savefig('grey_scaled.png')
plt.show()
