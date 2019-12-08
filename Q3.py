import cv2
import numpy as np
import matplotlib.pyplot as plt

class Q3:
    def __init__(self):
        self.img = cv2.imread('images/OriginalPerspective.png')
        self.rect = np.zeros((4, 2), dtype = "float32")
        self.count =0
   
            
    def trans_image(self, scale, t_x, t_y, angle):
        
        try:
            scale = float(scale)
            t_x = int(t_x)
            t_y = int(t_y)
            angle = int(angle)
        except:
            print('input error')
            return
        
        
        ori_trans_img = cv2.imread('images/OriginalTransform.png')
        # scale
        image = cv2.resize(ori_trans_img,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
        
        # translate
        rows,cols, c = ori_trans_img.shape
        M = np.float32([[1,0,t_x],[0,1,t_y]])
        image = cv2.warpAffine(image,M,(cols,rows))
        
        #rotate
        M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),angle,1)
        image = cv2.warpAffine(image,M,(cols,rows))
        
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()
        
  


