from keras import models
from keras import layers
from keras import optimizers
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import tools

epchos = 50
steps_per_epoch = 1000
batch_size = 50
model_save_path = "model/cifar.h5"

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=20,
    horizontal_flip=True,
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer=optimizers.RMSprop(1e-4, decay=1e-6),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(
    train_datagen.flow(train_images, train_labels, batch_size=batch_size), 
    steps_per_epoch=steps_per_epoch,
    epochs=epchos,
    validation_data=test_datagen.flow(test_images, test_labels, batch_size=batch_size),
    validation_steps=200,
    workers=4)

model.save(model_save_path)

tools.show_accuracy(history)
tools.show_loss(history)
