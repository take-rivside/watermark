import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('sample.png',0)

dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)

dft_shift = np.fft.fftshift(dft)


img2 = cv2.imread("takeru1.png", 0)    #埋め込む画像の読み込み
print(dft_shift.shape)
h,w,c = dft_shift.shape
img2 = cv2.resize(img2, (w,h))

dft_shift[img2 != 255] = 0

f_ishift = np.fft.ifftshift(dft_shift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

img_back /= 146535;

print(img_back[10,10]);

cv2.imwrite('a4.png',img_back )
