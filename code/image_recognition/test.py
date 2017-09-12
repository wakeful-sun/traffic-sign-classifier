import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('./test_signs/00000_00010.ppm')
r = cv2.resize(img, (32,32))
plt.imshow(r)
plt.show()
print(r.shape)
