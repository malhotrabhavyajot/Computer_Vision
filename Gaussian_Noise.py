
import numpy as np
import cv2

img = cv2.imread('grey_scaled.png')
# Generate Gaussian noise
gauss = np.random.normal(0, 1, img.size)
gauss = gauss.reshape(img.shape[0], img.shape[1], img.shape[2]).astype('uint8')
# Add the Gaussian noise to the image
img_gauss = cv2.add(img, gauss)
# Display the image

cv2.imwrite('Gaussian_Noise.png', img_gauss)
cv2.imshow('Grey Scaled image with Gaussian Noise', img_gauss)
cv2.waitKey(0)
