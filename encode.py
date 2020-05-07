import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("sample.png")

gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

dft = cv2.dft(np.float32(gray),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

#dft =cv2.resize(dft,(2133,2133))

#print(img2.shape)
#print(dft_shift.shape)




img2 = cv2.imread("takeru1.png")

#img2 = cv2.threshold(img1, 128, 255, cv2.THRESH_BINARY)

#img2 = 20*np.log(cv2.magnitude(img2[:,:,0],img2[:,:,1]))

h,w,c = dft_shift.shape
img2 = cv2.resize(img2, (w,h))
dft_shift = cv2.resize(dft_shift, (w,h))
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_RGBA2GRAY)
#dft_shift = cv2.resize(dft_shift, (w,h))


print(dft_shift.shape)
for x in range(w):
    for y in range(h):
        b, g, r = img2[x,y]
        if 0<=b<=100 and 0<=g<=100 and 0<=r<=100:
            dft_shift[x,y] = 0

#magnitude_spectrumウォーターマーク入りスペクトル
#magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
#plt.imshow(magnitude_spectrum, cmap = 'gray')
#plt.show()


#img_backウォーターマーク入りグレースケール
fshift=dft_shift
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
#plt.subplot(121),plt.imshow(img, cmap = 'gray')
#plt.title('Input Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
#plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
#plt.show()

plt.imshow(img_back, cmap = 'gray')
plt.xticks([]), plt.yticks([])
plt.show()
#plt.savefig("g.png")

#decode
#f = np.fft.fft2(img_back)
#fshift = np.fft.fftshift(f)
#magnitude_spectrum = 20*np.log(np.abs(fshift))

#plt.imshow(magnitude_spectrum, cmap = 'gray')
#plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
#plt.show()
