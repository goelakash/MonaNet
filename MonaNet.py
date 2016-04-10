# import theano
# import theano.tensor as T
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from random import randint

image_file = cv2.imread("Mona Lisa.jpg")

x1 = []
x2 = []
y1 = []
y2 = []

for i in xrange(image_file.shape[0]):
    for j in xrange(image_file.shape[1]):
        prob = randint(0,9)
        if prob>0:
            # np.insert(X_train,[[i,j]], axis=0)
            # np.insert(Y_train,[image_file[i,j,:].astype('float32')/255], axis=0)

            x1.append([i,j])
            y1.append(image_file[i,j,:].astype('float32')/255)
        else:
            # np.insert(X_test,[[i,j]], axis=0)
            # np.insert(Y_test,[image_file[i,j,:].astype('float32')/255], axis=0)

            x2.append([i,j])
            y2.append(image_file[i,j,:].astype('float32')/255)


X_train = np.array(x1)
X_test = np.array(x2)
Y_train = np.array(y1)
Y_test = np.array(y2)

print len(X_train), len(Y_train), len(X_test), len(Y_test)

model = Sequential()
model.add(Dense(5, input_dim=2, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(50, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(X_train, Y_train,
          nb_epoch=20,
          batch_size=400,
          show_accuracy=True)

score = model.evaluate(X_test, Y_test, batch_size=400)
