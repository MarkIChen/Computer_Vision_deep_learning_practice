import matplotlib.pyplot as plt
import cv2
import numpy as np
import math


class Q4:
    def __init__(self):
        self.school = cv2.imread('images/School.jpg')
        self.gray_school = cv2.cvtColor(self.school, cv2.COLOR_BGR2GRAY)
        
        # Filters
        self.Gaussian_kernel = np.array((
                [1, 2, 1],
                [2, 4, 2],
                [1, 2, 1]), dtype='float') *(1.0/16)
        self.sobelX = np.array((
        	[-1, 0, 1],
        	[-2, 0, 2],
        	[-1, 0, 1]), dtype="int")
        self.sobelY = np.array((
        	[-1, -2, -1],
        	[0, 0, 0],
        	[1, 2, 1]), dtype="int")
        

    def showGaussian(self):
        print('calculating')
        result =  self.convolve(self.gray_school, self.Gaussian_kernel)
        plt.imshow(result, cmap='gray')
        plt.show()
        
        
    def shoowSobelX(self):
        result =  self.convolve(self.gray_school, self.sobelX)
        plt.imshow(result, cmap='gray')
        plt.show()
    
    def shoowSobelY(self):
        result =  self.convolve(self.gray_school, self.sobelY)
        plt.imshow(result, cmap='gray')
        plt.show()
    
    def shoowManitude(self):
        gau_img =  self.convolve(self.gray_school, self.Gaussian_kernel)
        sobelX_result = self.convolve(gau_img, self.sobelX)
        sobelY_result = self.convolve(gau_img, self.sobelY)
        result =  self.magnitude(sobelX_result, sobelY_result)
        plt.imshow(result, cmap='gray')
        plt.show()
        

    def convolve(self, image, kernel):
    
        (iH, iW) = image.shape[:2]
        (kH, kW) = kernel.shape[:2]
    
        pad = (kW - 1)//2
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    
        output = np.zeros((iH, iW), dtype="float32")
        
        for y in np.arange(pad, iH+pad):
            for x in np.arange(pad, iW+pad):
                roi = image[y-pad : y + pad + 1, x-pad : x+pad+1]
                k = (roi * kernel).sum()
                output[y - pad, x - pad] = k
        
        output = self.normalize(output)
    
        return output
    
    def normalize(self, input_img):
        width = len(input_img)
        height = len(input_img[0])
        output=np.zeros((width, height))
        for i in range(len(input_img)):
            for j in range(len(input_img[i])):
                if(input_img[i][j] < 0 ):
                    input_img[i][j] = 0
                elif (input_img[i][j] > 255 ):
                    input_img[i][j] = 255
                   
        data_max = int(np.amax(input_img))
        data_min = int(np.amin(input_img))
        delta = data_max - data_min
        
        for i in range(len(input_img)):
            for j in range(len(input_img[i])):
                output[i][j] = int(255 * ((input_img[i][j]) - data_min) / delta)
    
        output = output.astype("uint8")    
        return output
               
    def magnitude(self, img1, img2):
        width = len(img1)
        height = len(img1[0])
        output=np.zeros((width, height))
    
        for i in range(width):
            for j in range(height):
                x  = int(img1[i][j])
                y  = int(img2[i][j])
                output[i][j] = int(math.sqrt( x*x+ y*y ))
    
        return output.astype("uint8")



