from keras.utils import to_categorical
from keras import layers
from keras import models
from keras import regularizers
from sklearn.model_selection import train_test_split
from simpledatasetloader import SimpleDatasetLoader
import numpy as np
from matplotlib import pylab as pl

dloader = SimpleDatasetLoader()

(data_x, data_y) = dloader.load('../SMILEsmileD/SMILEs/positives/positives7',
                                '../SMILEsmileD/SMILEs/positives/laplacian', 1)

dat = []
N = data_y.shape[0]
for n in range(N):
    dat.append(data_y[n].reshape((64*64)))
data_y = np.array(dat)
del dat

(train_x, test_x, train_y, test_y) = train_test_split(data_x, data_y, test_size=0.40)


model = models.Sequential()
#kernel_regularizer = regularizers.l2(10.1),
model.add(layers.Conv2D(10, (3,3), activation='relu', input_shape=(64, 64, 1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(20, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(20, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(64*64))

model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=100, batch_size=64)

y = model.predict(test_x[0].reshape(1,64,64,1))

pl.figure()
pl.imshow(test_x[0].reshape(64,64), cmap=pl.cm.gray)
pl.figure()
pl.imshow(test_y[0].reshape(64,64), cmap=pl.cm.gray)
pl.figure()
pl.imshow(y.reshape(64,64), cmap=pl.cm.gray)
pl.show()