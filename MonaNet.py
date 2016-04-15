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
            x1.append([i,j])
            y1.append(image_file[i,j,:].astype('float32')/255.0)
        else:
            x2.append([i,j])
            y2.append(image_file[i,j,:].astype('float32')/255.0)

X_train = np.array(x1)
Y_train = np.array(y1)
X_test = np.array(x2)
Y_test = np.array(y2)

X = []
X.append(X_train)
X.append(X_test)

Y = []
Y.append(Y_train)
Y.append(Y_test)

model = Sequential()
model.add(Dense(300, input_dim=2, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(300, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(300, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(300, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(300, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3, init='uniform'))
model.add(Activation('linear'))
model.add(Dropout(0.5))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(X_train, Y_train,
          nb_epoch=20,
          batch_size=400,
          show_accuracy=True)

score = model.evaluate(X_test, Y_test, batch_size=400)

output_image = {}

for i in X:
    output_image[i] = model.predict(i)

print output_image
