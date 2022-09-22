from os import listdir
from os.path import isfile, join, isdir
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import numpy as np
import PIL
import tensorflow as tf
import os

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from keras import models
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.datasets import mnist


HEIGHT = 256
WIDTH = 256

labels = []
images = []

class Image():
  def __init__(self, imagename, image):
    self.imagename = imagename
    self.oldimagename = imagename
    self.changed = False
    self.image = image


def getImages():
  dirs = [f for f in listdir("flowers") if isdir(join("flowers", f))]
  for dir in dirs:
    print(dir)
    labels.append(dir)
    files = [f for f in listdir(join("flowers", dir)) if isfile(join(join("flowers", dir), f))]
    for file in files:
      filename = join(join("flowers", dir), file)
      images.append(Image(dir, cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)))
  return images

def splitTrainVal(images, split):
  imagerangelist = list(range(len(images)))
  imageindices = random.sample(imagerangelist, int(len(imagerangelist)*(1-split)))
  set1 = [images[i] for i in imageindices]
  set2 = []
  for i in imagerangelist:
    if not(i in imageindices):
      set2.append(images[i])
  return set1, set2

def showSample(images, samplesize):
  fig = plt.figure(figsize=(12, 8))

  rows = math.floor(samplesize**0.5)
  columns = math.ceil(samplesize/math.floor(samplesize**0.5))

  sample = random.sample(images, samplesize)
  for index, image in enumerate(sample):
    fig.add_subplot(rows, columns, index+1)
      
    # showing image
    plt.imshow(image.image)
    plt.axis('off')
    plt.title(image.imagename, y=-0.2)

  plt.show()

def swapLabels(images, proportion):
  changeimages = random.sample(list(enumerate(images)), round(len(images)*proportion))
  for index, image in changeimages:
    oldlabel = image.imagename
    newlabellist = labels[:]
    newlabellist.remove(oldlabel)
    newlabel = random.choice(newlabellist)
    image.imagename = newlabel
    images[index] = image
  return images, [i[1] for i in changeimages]

def splitBuildingSet(images):
  imagerangelist = list(range(len(images)))
  imageindices = random.sample(imagerangelist, len(imagerangelist)//2)
  set1 = [images[i] for i in imageindices]
  set2 = []
  for i in imagerangelist:
    if not(i in imageindices):
      set2.append(images[i])
  return set1, set2

def showImage(image, model):
  predictions = model.predict(tf.expand_dims(((cv2.resize(image.image, (HEIGHT, WIDTH)))), 0))
  score = tf.nn.softmax(predictions[0])
  print(score)
  print(labels[np.argmax(score)])
  print(labels[np.argmax(score)] == image.oldimagename)
  cv2.imshow("win", image.image)
  print(image.oldimagename)


  

images = random.sample(getImages(), 700)
images, val_images = splitTrainVal(images, 0.5)
print(len(images), len(val_images))
#showSample(images, 16)
#showSample(val_images, 16)
images, changes = swapLabels(images, 0)
#showSample(changes, 16)
set1, set2 = splitBuildingSet(images)

imagelist = tf.constant([cv2.resize(i.image, (WIDTH, HEIGHT)) for i in set1]) # or tf.convert_to_tensor(files)
imagelabels = tf.constant([labels.index(i.imagename) for i in set1]) # or tf.convert_to_tensor(labels)
dataset = tf.data.Dataset.from_tensor_slices((imagelist, imagelabels))

imagelisttest = tf.constant([cv2.resize(i.image, (WIDTH, HEIGHT)) for i in val_images]) # or tf.convert_to_tensor(files)
imagelabelstest = tf.constant([labels.index(i.imagename) for i in val_images]) # or tf.convert_to_tensor(labels)
datasettest = tf.data.Dataset.from_tensor_slices((imagelisttest, imagelabelstest))

BATCH_SIZE = 1
SHUFFLE_BUFFER_SIZE = 100

train_dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = datasettest.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(HEIGHT, WIDTH, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(len(labels))
])

model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])


checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


model.fit(train_dataset, epochs=10, callbacks=[cp_callback])
model.load_weights(checkpoint_path)

for i in random.sample(val_images, 10):
  showImage(i, model)

#cv2.waitKey(0)

print("Evaluate on test data")
results = model.evaluate(test_dataset, batch_size=1)
print("test loss, test acc:", results)
