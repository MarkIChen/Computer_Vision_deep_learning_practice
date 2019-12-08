import cv2
import numpy as np


img = cv2.imread('images/OriginalPerspective.png')
rect = np.zeros((4, 2), dtype = "float32")
count =0

def four_point_transform(image, pts):
	(tl, tr, br, bl) = rect
 
	dst = np.array([
		[0, 0],
		[430, 0],
		[430, 430],
		[0, 430]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (430, 430))
 
	# return the warped image
	return warped

def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY, rect, count
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),10,(0,0,255),-1)
        cv2.imshow('images',img)
        print('x = %d, y = %d'%(x, y))
        mouseX,mouseY = x,y
        rect[count][0] = x
        rect[count][1] = y
        count+=1


while(1):
    cv2.namedWindow('images',  cv2.WINDOW_NORMAL)
    cv2.imshow('images',img)

    cv2.setMouseCallback('images',draw_circle)
    if(count == 4):
        break
    
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('a'):
        print(mouseX,mouseY)
   

print(rect)
transformed = four_point_transform(img, rect)
cv2.namedWindow('transformed',  cv2.WINDOW_NORMAL)
cv2.imshow("transformed", transformed)
cv2.waitKey(0)



