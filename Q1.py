import cv2
#import numpy as np
import matplotlib.pyplot as plt

#Q1-1 show dog image

    
def showImg():
    dog_img = cv2.imread('images/dog.bmp')
    
    height, width, channels = dog_img.shape
    
    print('Height: %d' % height)
    print('Width: %d' % width)
    
    plt.imshow(cv2.cvtColor(dog_img, cv2.COLOR_BGR2RGB))
    plt.show()
    
def convertColor():
    color_img = cv2.imread('images/color.png')
    b,g,r = cv2.split(color_img)
    color_convert = cv2.merge([g, r, b])
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(color_convert, cv2.COLOR_BGR2RGB))
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def flip():
    dog_img = cv2.imread('images/dog.bmp')
    flipHorizontal = cv2.flip(dog_img, 1)
    
    plt.imshow(cv2.cvtColor(flipHorizontal, cv2.COLOR_BGR2RGB))
    plt.show()
