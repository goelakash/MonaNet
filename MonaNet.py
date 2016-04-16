import sys
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop
from random import randint, shuffle

image_file = cv2.imread("Mona Lisa.jpg")

h = image_file.shape[0]
w = image_file.shape[1]

model = Sequential()

model.add(Dense(500, input_dim=2, init='uniform'))
model.add(Activation('relu'))

model.add(Dense(500, init='uniform'))
model.add(Activation('relu'))

model.add(Dense(500, init='uniform'))
model.add(Activation('relu'))

model.add(Dense(500, init='uniform'))
model.add(Activation('relu'))

model.add(Dense(500, init='uniform'))
model.add(Activation('relu'))

model.add(Dense(3, init='uniform'))
model.add(Activation('linear'))

# sgd = SGD(lr=0.1, momentum=0.94)
model.compile(loss='mean_squared_error', optimizer=RMSprop())

x1 = []
y1 = []

for i in xrange(h):
    for j in xrange(w):
        x1.append([i,j])
        y1.append(image_file[i,j,:].astype('float32')/255.0)

zip_1 = [i for i in zip(x1,y1)]

nb_epochs = 1000

for e in xrange(nb_epochs):

    
    print "Epoch: ",e

    shuffle(zip_1)

    X_train = np.array([i for (i,j) in zip_1])
    Y_train = np.array([j for (i,j) in zip_1])
    
    model.fit(X_train, Y_train,
          nb_epoch=1,
          batch_size=500,
          show_accuracy=True)



    X = []
    X.extend(x1)
    X = sorted(X)

    predictions = model.predict(np.array(X))*255.0
    for i in xrange(len(predictions)):
        for j in xrange(3):
            predictions[i][j] = min(predictions[i][j],255.0)
    predictions = predictions.astype('uint8')
    
    output_image = []
    index = 0

    for i in xrange(h):
        row = []

        for j in xrange(w):
            row.append(predictions[index])
            index += 1

        row = np.array(row)
        output_image.append(row)

    output_image = np.array(output_image)
    cv2.imwrite('out_mona_500x5_e'+str(e)+'.png',output_image)