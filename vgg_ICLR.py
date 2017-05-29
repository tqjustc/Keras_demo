
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D

model = Sequential()
# layer 1
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', input_shape=(3, 224, 224), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
# layer 2
model.add(Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
# layer 3
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))
# layer 4
model.add(Conv2D(filters=512, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
model.add(Conv2D(filters=512, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
model.add(Conv2D(filters=512, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))
# layer 5
model.add(Conv2D(filters=512, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
model.add(Conv2D(filters=512, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
model.add(Conv2D(filters=512, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))
# layer 6
model.add(Dense(units=4096, activation='relu'))
model.add(Dense(units=4096, activation='relu'))
model.add(Dense(units=1000, activation='softmax'))





sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-5, nesterov=True)

model.compile(
    optimizer=sgd,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

