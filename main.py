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

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


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
  

images = getImages()
showSample(images, 16)
images, changes = swapLabels(images, 0.2)
showSample(changes, 16)
set1, set2 = splitBuildingSet(images)
print(len(set1), len(set2))