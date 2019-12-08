import gzip, cv2
import numpy as np
from random import randint
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import TensorBoard

import matplotlib.pyplot as plt

class Q5():
    
    def __init__(self):
        self.train = {}
        self.test = {}
        self.BATCH_SIZE = 128
        self.learning_rate=0.01
        self.train['features'], self.train['labels'] = self.read_mnist('dataset/train-images-idx3-ubyte.gz', 'dataset/train-labels-idx1-ubyte.gz')
        self.test['features'], self.test['labels'] = self.read_mnist('dataset/t10k-images-idx3-ubyte.gz', 'dataset/t10k-labels-idx1-ubyte.gz')
        self.validation = {}
        self.train['features'], self.validation['features'], self.train['labels'], self.validation['labels'] = train_test_split(self.train['features'], self.train['labels'], test_size=0.2, random_state=0)

        # Pad images with 0s
        self.train['features']      = np.pad(self.train['features'], ((0,0),(2,2),(2,2),(0,0)), 'constant')
        self.validation['features'] = np.pad(self.validation['features'], ((0,0),(2,2),(2,2),(0,0)), 'constant')
        self.test['features']       = np.pad(self.test['features'], ((0,0),(2,2),(2,2),(0,0)), 'constant')
  
        # set up LeNet architecture
        self.model = keras.Sequential()
        
        self.model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32,32,1)))
        self.model.add(layers.AveragePooling2D())
        self.model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
        self.model.add(layers.AveragePooling2D())
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(units=120, activation='relu'))
        self.model.add(layers.Dense(units=84, activation='relu'))
        self.model.add(layers.Dense(units=10, activation = 'softmax'))
        self.model.summary()
        
        sgd = optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer= sgd, metrics=['accuracy'])
        
        self.X_train, self.y_train = self.train['features'], to_categorical(self.train['labels'])
        self.X_validation, self.y_validation = self.validation['features'], to_categorical(self.validation['labels'])
        
        self.train_generator = ImageDataGenerator().flow(self.X_train, self.y_train, batch_size=self.BATCH_SIZE)
        self.validation_generator = ImageDataGenerator().flow(self.X_validation, self.y_validation, batch_size=self.BATCH_SIZE)
        
        self.steps_per_epoch = self.X_train.shape[0]//self.BATCH_SIZE
        self.validation_steps = self.X_validation.shape[0]//self.BATCH_SIZE


        
    def showTrainImg(self):
        train_size = self.train['features'].shape[0] - 1

        for i in range(0, 9):
            position = randint(0, train_size)
            plt.subplot(250 + 1 + i)
            plt.imshow(self.train['features'][position].squeeze(), cmap=plt.get_cmap('gray'))
            plt.title('Example %d\n Label: %d' % (position, self.train['labels'][position]), fontsize=6)
        
        position = randint(0, train_size)
        plt.subplot(2, 5, 10)
        plt.imshow(self.train['features'][position].squeeze(), cmap=plt.get_cmap('gray'))
        plt.title('Example %d\n Label: %d' % (position, self.train['labels'][position]), fontsize=6)
            
        plt.show()
    
    def showParameter(self):
        print('hyperparameter:')
        print('batch size: %d' % self.BATCH_SIZE)
        print('learning rate: %f' % self.learning_rate)
        print('optmizer: SGD')
        
    def trainOneEpoch(self):
        histories_per_epoch = Histories()

        self.model.fit_generator(self.train_generator, steps_per_epoch=self.steps_per_epoch, epochs= 1, 
                    validation_data=self.validation_generator, validation_steps=self.validation_steps, 
                    shuffle=True, callbacks=[histories_per_epoch])

        plt.plot(histories_per_epoch.losses[:self.X_train.shape[0]//self.BATCH_SIZE])
        plt.title('model loss per epoch')
        plt.ylabel('loss')
        plt.xlabel('iteration')
        plt.show()
        
    def showTrainResult(self):
        plt.subplot(121)
        img_acc = cv2.imread('train_result/accuracy.png')
        plt.imshow(cv2.cvtColor(img_acc, cv2.COLOR_BGR2RGB))
        
        plt.subplot(122)
        img_loss = cv2.imread('train_result/loss.png')
        plt.imshow(cv2.cvtColor(img_loss, cv2.COLOR_BGR2RGB))
        
        plt.show()
    
    def evaluateIndex(self, index):
 
        try:
            index = int(index)
        except:
            print('input error')
            return
        
        trained_model = load_model('train_result/model_50.h5')
        image = self.test['features'][index].squeeze()
        plt.subplot(2, 1, 1)
        plt.title('Example %d. Label: %d' % (index, self.test['labels'][index]))
        plt.imshow(image, cmap=plt.cm.gray_r)
        x = np.expand_dims(self.test['features'][index], axis=0)
        score = trained_model.predict(x.astype(np.float32))
        
        x_label = np.arange(10)
        x_label
        plt.subplot(2, 1, 2)
        plt.bar(x_label, score[0])
        plt.xticks(x_label)
        plt.show()
    
        
    def read_mnist(self, images_path: str, labels_path: str):
        with gzip.open(labels_path, 'rb') as labelsFile:
            labels = np.frombuffer(labelsFile.read(), dtype=np.uint8, offset=8)
    
        with gzip.open(images_path,'rb') as imagesFile:
            length = len(labels)
            # Load flat 28x28 px images (784 px), and convert them to 28x28 px
            features = np.frombuffer(imagesFile.read(), dtype=np.uint8, offset=16) \
                            .reshape(length, 784) \
                            .reshape(length, 28, 28, 1)
            
        return features, labels
        
class Histories(Callback):

    def on_train_begin(self,logs={}):
        self.losses = []
        self.accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))
    

q5 = Q5()
q5.evaluateIndex(1)
