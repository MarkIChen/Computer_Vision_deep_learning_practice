import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow
import cv2
import Q1, Q2, Q4, Q5, Q3
import matplotlib.pyplot as plt

class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(50, 50, 1000, 600)
        self.setWindowTitle("OpenCV DL HW1")
        self.q4 = Q4.Q4()
        self.q5 = Q5.Q5()
        self.q3  = Q3.Q3()
        self.initUI()
#        self.quit_btn()
        
    def initUI(self):
        self.label = QtWidgets.QLabel(self)
        self.label.setText('1. Image Process')
        self.label.move(10, 5)
        
        self.q1_b1 = QtWidgets.QPushButton(self)
        self.q1_b1.setText('1.1 Load Image')
        self.q1_b1.move(10, 30)
        self.q1_b1.resize(150, 30)
        self.q1_b2 = QtWidgets.QPushButton(self)
        self.q1_b2.setText('1.2 Color Coversion')
        self.q1_b2.move(10, 60)
        self.q1_b2.resize(150, 30)
        self.q1_b3 = QtWidgets.QPushButton(self)
        self.q1_b3.setText('1.3 Image Flipping')
        self.q1_b3.move(10, 90)
        self.q1_b3.resize(150, 30)
        self.q1_b4 = QtWidgets.QPushButton(self)
        self.q1_b4.setText('1.4 Blending')
        self.q1_b4.move(10, 120)
        self.q1_b4.resize(150, 30)
        
        self.q2_label = QtWidgets.QLabel(self)
        self.q2_label.setText('2. Adaptive Threshold')
        self.q2_label.move(200, 5)
        self.q2_b1 = QtWidgets.QPushButton(self)
        self.q2_b1.setText('2.1 Global Threshold')
        self.q2_b1.move(200, 30)
        self.q2_b1.resize(150, 30)
        self.q2_b2 = QtWidgets.QPushButton(self)
        self.q2_b2.setText('2.2 Local Threshold')
        self.q2_b2.move(200, 60)
        self.q2_b2.resize(150, 30)

        self.q4_label = QtWidgets.QLabel(self)
        self.q4_label.setText('4. Convolution')
        self.q4_label.move(200, 100)
        self.q4_b1 = QtWidgets.QPushButton(self)
        self.q4_b1.setText('4.1 Gaussian')
        self.q4_b1.move(200, 130)
        self.q4_b1.resize(150, 30)
        self.q4_b2 = QtWidgets.QPushButton(self)
        self.q4_b2.setText('4.2 SobelX')
        self.q4_b2.move(200, 160)
        self.q4_b2.resize(150, 30)
        self.q4_b3 = QtWidgets.QPushButton(self)
        self.q4_b3.setText('4.3 SobelY')
        self.q4_b3.move(200, 190)
        self.q4_b3.resize(150, 30)
        self.q4_b4 = QtWidgets.QPushButton(self)
        self.q4_b4.setText('4.4 Magnitude')
        self.q4_b4.move(200, 220)
        self.q4_b4.resize(150, 30)
        
        self.q3_label = QtWidgets.QLabel(self)
        self.q3_label.setText('3. Image Tranformation')
        self.q3_label.move(400, 5)
        
        self.q3_ang_label = QtWidgets.QLabel(self)
        self.q3_ang_label.setText('Angle')
        self.q3_ang_label.move(400, 40)
        self.q3_ang_textbox = QtWidgets.QLineEdit(self)
        self.q3_ang_textbox.move(450, 50)
        self.q3_ang_textbox.resize(100, 20)
        self.q3_scale_label = QtWidgets.QLabel(self)
        self.q3_scale_label.setText('scale')
        self.q3_scale_label.move(400, 70)
        self.q3_scale_textbox = QtWidgets.QLineEdit(self)
        self.q3_scale_textbox.move(450, 80)
        self.q3_scale_textbox.resize(100, 20)
        self.q3_tX_label = QtWidgets.QLabel(self)
        self.q3_tX_label.setText('t_x')
        self.q3_tX_label.move(400, 100)
        self.q3_tX_textbox = QtWidgets.QLineEdit(self)
        self.q3_tX_textbox.move(450, 110)
        self.q3_tX_textbox.resize(100, 20)
        self.q3_tY_label = QtWidgets.QLabel(self)
        self.q3_tY_label.setText('t_y')
        self.q3_tY_label.move(400, 130)
        self.q3_tY_textbox = QtWidgets.QLineEdit(self)
        self.q3_tY_textbox.move(450, 140)
        self.q3_tY_textbox.resize(100, 20)
        
        self.q3_b1 = QtWidgets.QPushButton(self)
        self.q3_b1.setText('3.1 Rotate, Scale, translation')
        self.q3_b1.move(400, 200)
        self.q3_b1.resize(200, 30)
        self.q3_b2 = QtWidgets.QPushButton(self)
        self.q3_b2.setText('3.2 perspective transform')
        self.q3_b2.move(400, 230)
        self.q3_b2.resize(200, 30)
        
        self.q5_label = QtWidgets.QLabel(self)
        self.q5_label.setText('5. Training MNIST Classifier Using LeNet5')
        self.q5_label.move(650, 5)
        self.q5_b1 = QtWidgets.QPushButton(self)
        self.q5_b1.setText('5.1 Show Image')
        self.q5_b1.move(650, 30)
        self.q5_b1.resize(200, 30)
        self.q5_b2 = QtWidgets.QPushButton(self)
        self.q5_b2.setText('5.2 Show Hpyerparameter')
        self.q5_b2.move(650, 60)
        self.q5_b2.resize(200, 30)
        self.q5_b3 = QtWidgets.QPushButton(self)
        self.q5_b3.setText('5.3 Train 1 epoch')
        self.q5_b3.move(650, 90)
        self.q5_b3.resize(200, 30)
        self.q5_b4 = QtWidgets.QPushButton(self)
        self.q5_b4.setText('5.4 Show Training Result')
        self.q5_b4.move(650, 120)
        self.q5_b4.resize(200, 30)
        
        self.q5_textbox = QtWidgets.QLineEdit(self)
        self.q5_textbox.move(660, 160)
        self.q5_textbox.resize(100, 20)
        self.q5_textbox.setPlaceholderText('input 0~9999')
        
        self.q5_b5 = QtWidgets.QPushButton(self)
        self.q5_b5.setText('5.5 Inference')
        self.q5_b5.move(650, 180)
        self.q5_b5.resize(200, 30)

        self.q1_b1.clicked.connect(Q1.showImg)
        self.q1_b2.clicked.connect(Q1.convertColor)
        self.q1_b3.clicked.connect(Q1.flip)
        
        self.q2_b1.clicked.connect(Q2.showGlobalThresh)
        self.q2_b2.clicked.connect(Q2.showLocalThresh)
  
        self.q4_b1.clicked.connect(self.q4.showGaussian)
        self.q4_b2.clicked.connect(self.q4.shoowSobelX)
        self.q4_b3.clicked.connect(self.q4.shoowSobelY)
        self.q4_b4.clicked.connect(self.q4.shoowManitude)
        
        self.q5_b1.clicked.connect(self.q5.showTrainImg)
        self.q5_b2.clicked.connect(self.q5.showParameter)
        self.q5_b3.clicked.connect(self.q5.trainOneEpoch)
        self.q5_b4.clicked.connect(self.q5.showTrainResult)
        self.q5_b5.clicked.connect(self.q5_5_clicked)
        
        self.q3_b1.clicked.connect(self.q3_1_clicked)
        self.q3_b2.clicked.connect(self.q3_2_clicked)
        
    def q5_5_clicked(self):
        self.q5.evaluateIndex(self.q5_textbox.text())
    
    def q3_1_clicked(self):
        self.q3.trans_image(self.q3_scale_textbox.text(), self.q3_tX_textbox.text(), self.q3_tY_textbox.text(), self.q3_ang_textbox.text())

    def q3_2_clicked(self):
        print('Please run perspective_tranform.py file to test this function.')


def displayWindow():
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())

    
displayWindow()



