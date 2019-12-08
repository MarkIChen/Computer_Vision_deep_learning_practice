import cv2
import matplotlib.pyplot as plt


def showGlobalThresh():
    image = cv2.imread('images/QR.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret,thresh = cv2.threshold(image,90,255,cv2.THRESH_BINARY)
    plt.imshow(thresh, cmap='gray')
    plt.show()
    
def showLocalThresh():
    image = cv2.imread('images/QR.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,19,1)
    
    plt.imshow(thresh, cmap='gray')
    plt.show()
    