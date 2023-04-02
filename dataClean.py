from os import listdir
from os.path import isfile, join, isdir
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import numpy as np
import tensorflow as tf
import os
import shutil
import time
import datetime
import IPython
from contextlib import redirect_stdout
from matplotlib import pyplot as plt
import Rouse.modelResNet as modelResNet
import copy

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from keras import models
from keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from keras.preprocessing import image
try:
  from google.colab.patches import cv2_imshow
except:
  pass

HEIGHT = 32
WIDTH = 32
CHANNELS = 3
BATCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 100
TRAIN_EPOCHS = 120
SECONDARY_EPOCHS = 5
MAIN_EPOCHS = 4

"""if "display" not in globals():
  def display(*args, **kwargs):
    if len(args) == 0:
      return
    print(args[0])"""
if "cv2_imshow" not in globals():
  def cv2_imshow(img):
    cv2.imshow("window", img)

class CustomCallback(keras.callbacks.Callback):
    
    """def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))"""

    def on_epoch_begin(self, epoch, logs=None):
        global train_epoch
        train_epoch = epoch + 1
        loading_bar.display()

class LoadingBar():
  def __init__(self, verbose):
    self.structure = "| Elapsed: {} | Epoch {} | {} half epoch | Model training progress: {} {}/{} | Validation accuracy: {}% | Validation loss: {} | Dataset modification progress: {} {}/{} | Dataset half accuracy before half epoch: {}% | Dataset half accuracy after half epoch: {}% | Correct relabelling: {}% | Incorrect relabelling: {}% | Total dataset accuracy: {}% |"
    self.out = display(display_id=True)
    self.verbose = verbose
  
  def display(self, save=False):

    if not(self.verbose):
      return

    time_elapsed = str(datetime.timedelta(seconds=int(time.time()-epochtime)))

    current_epoch = str((epoch + 1)).rjust(len(str(MAIN_EPOCHS)))
    half_epoch = " First" if half == 0 else "Second"

    progress_dash_train = "=" * train_epoch + "." * (TRAIN_EPOCHS - train_epoch)
    epoch_train = str(train_epoch).rjust(len(str(TRAIN_EPOCHS)))
    epoch_train_total = TRAIN_EPOCHS

    val_accuracy_str = str(format(val_accuracy, ".2f") if val_accuracy != "?" else "?").rjust(5)
    val_loss_str = str(format(val_loss, ".3f") if val_loss != "?" else "?").rjust(5)

    dataset_modification_progress_dash_train = "=" * (dataset_modification_progress//(len(set2[0])//20)) + "." * (20 - (dataset_modification_progress//(len(set2[0])//20)))
    dataset_modification_progress_str = str(min(dataset_modification_progress, len(set2[0]))).rjust(len(str(len(set2[0]))))
    dataset_total_modifications_str = str(len(set2[0]))

    dataset_accuracy_before_str = str(format(dataset_accuracy_before, ".2f") if dataset_accuracy_before != "?" else "?").rjust(5)
    dataset_accuracy_after_str = str(format(dataset_accuracy_after, ".2f") if dataset_accuracy_after != "?" else "?").rjust(5)
    dataset_accuracy_increase = str(format(accuracy_increase, ".2f") if accuracy_increase != "?" else "?").rjust(5)
    dataset_accuracy_decrease = str(format(accuracy_decrease, ".2f") if accuracy_decrease != "?" else "?").rjust(5)

    total_accuracy_str = str(format(total_accuracy, ".2f") if total_accuracy != "?" else "?").rjust(5)

    filledstructure = self.structure.format(time_elapsed, current_epoch, half_epoch, progress_dash_train, epoch_train, epoch_train_total, val_accuracy_str, val_loss_str, dataset_modification_progress_dash_train, dataset_modification_progress_str, dataset_total_modifications_str, dataset_accuracy_before_str, dataset_accuracy_after_str, dataset_accuracy_increase, dataset_accuracy_decrease, total_accuracy_str)
    maxlen = len(filledstructure)
    
    if save:
      self.out.update(IPython.display.Markdown(''))
      print("="*maxlen)
      print(filledstructure)
      print("="*maxlen)
      self.out = display(display_id=True)
    
    else:
      self.out.update(IPython.display.Pretty("="*maxlen+"\n"+filledstructure+"\n"+"="*maxlen))


class Image():
  def __init__(self, imagename, image, filename=None):
    self.imagename = imagename
    self.oldimagename = imagename
    self.changed = False
    self.image = image
    self.filename = filename


def scheduler(epoch, lr):
    if epoch == 40 or epoch == 80:
        return lr * 0.1
    else:
        return lr

# ResNet34 architecture
def convertToUseful(x_train, y_train, x_test, y_test):
  mean_image = np.mean(x_train, axis=0)

  # Subtract the mean from the training and test sets
  x_train = x_train.astype(np.float32) - mean_image.astype(np.float32)
  x_test = x_test.astype(np.float32) - mean_image.astype(np.float32)

  # Data normalization
  x_train = x_train.astype('float32') / 255.0
  x_test = x_test.astype('float32') / 255.0

  # One-hot encoding of labels
  y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
  y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

  return (x_train, y_train), (x_test, y_test)

def getImages(path = "/content/dataset/dataset"):
  global labels
  images = []
  labels = []
  ignored = []#["glacier", "street"]
  dirs = sorted([f for f in listdir(path) if isdir(join(path, f))])
  for ignore in ignored:
    dirs.remove(ignore)
  for dir in dirs:
    print(dir)
    labels.append(dir)
    files = [f for f in listdir(join(path, dir)) if isfile(join(join(path, dir), f))]
    for file in files:
      filename = join(join(path, dir), file)
      try:
        images.append(Image(dir, (cv2.imread(filename) if CHANNELS != 1 else cv2.imread(filename, 0)), filename=filename))
      except:
        print("Failed at" + filename)
  return images

def splitTrainVal(images, split):
  imagerangelist = list(range(len(images)))
  imageindices = random.sample(imagerangelist, int(len(imagerangelist)*(1-split)))
  imageindices.sort()
  complement_imageindices = list(set(imagerangelist) - set(imageindices))
  complement_imageindices.sort()
  set1 = [images[i] for i in imageindices]
  set2 = [images[i] for i in complement_imageindices]
  return set1, set2

def showSample(x_test, y_test, samplesize):
  fig = plt.figure(figsize=(12, 8))

  rows = math.floor(samplesize**0.5)
  columns = math.ceil(samplesize/math.floor(samplesize**0.5))

  sample = random.sample(range(len(x_test)), samplesize)
  for index, imageindex in enumerate(sample):
    fig.add_subplot(rows, columns, index+1)
      
    # showing image
    plt.imshow(x_test[imageindex])
    plt.axis('off')
    plt.title(y_test[imageindex], y=-0.2)

  plt.show()

def swapLabels(labels, proportion):
  changelabels = random.sample(list(enumerate(labels)), round(len(labels)*proportion))
  changelabels.sort()
  print(labels.shape)
  labellist = list(set(labels.flatten()))
  newlabels = copy.deepcopy(labels)
  for index, label in changelabels:
    oldlabel = label
    newlabellist = labellist.copy()
    newlabellist.remove(oldlabel.item())
    newlabel = random.choice(newlabellist)
    newlabels[index] = newlabel
  return newlabels, labels

def splitBuildingSet(x_train, y_train, y_train_old, proportion):
  p = len(x_train)

  # generate an array of random indices for splitting the data
  indices = np.random.choice(p, int(proportion*p), replace=False)

  # split each array into two sets using the same indices
  x_train1, x_train2 = x_train[indices], x_train[np.setdiff1d(np.arange(p), indices)]
  y_train1, y_train2 = y_train[indices], y_train[np.setdiff1d(np.arange(p), indices)]
  y_train_old1, y_train_old2 = y_train_old[indices], y_train_old[np.setdiff1d(np.arange(p), indices)]
  set1 = [x_train1, y_train1, y_train_old1]
  set2 = [x_train2, y_train2, y_train_old2]
  return set1, set2


def showImage(image, model):
  predictions = model.predict(tf.expand_dims(((cv2.resize(image.image, (HEIGHT, WIDTH)))), 0)/255)
  score = tf.nn.softmax(predictions[0])
  print(score)
  print(labels[np.argmax(score)])
  print(labels[np.argmax(score)] == image.oldimagename)
  cv2_imshow(image.image)
  print(image.oldimagename)

def saveSet(imageSet, path):
  try:
    shutil.rmtree(path)
    os.mkdir(path)
  except Exception as e:
    os.mkdir(path)
  for label in labels:
    try:
      os.mkdir(path+"/"+label)
    except Exception as e:
      print(e)
      pass
  try:
    os.rmdir(path+"/"+".ipynb_checkpoints")
  except:
    pass
  count = 0
  for image in imageSet:
    count += 1
    cv2.imwrite(path+"/"+image.imagename+"/"+str(count).zfill(8)+".jpg", image.image)


def getClassProportion(classlist):
  counts = []
  count = 0
  last = ""
  for i in classlist:
    if i != last:
      counts.append(count)
      count = 0
      last = i
    count += 1
  counts.append(count)
  return counts[1:]

def getLabelingAccuracy(labels, oldlabels):
  total = len(labels)
  num_matches = np.sum(labels == oldlabels)
  return num_matches/total*100

def trainModel(ds, val_ds):
  print(ds)
  num_classes = 10

  """data_augmentation = keras.Sequential(
    [
      layers.RandomFlip("horizontal",
                        input_shape=(HEIGHT,
                                    WIDTH,
                                    CHANNELS)),
      layers.RandomRotation(0.1),
      layers.RandomZoom(0.1),
    ]
  )"""


  
  model = modelResNet.createModel(inputshape=(HEIGHT, WIDTH, CHANNELS), outputclasses=num_classes)

  checkpoint_path = "training_1/cp.ckpt"
  checkpoint_dir = os.path.dirname(checkpoint_path)

  # Create a callback that saves the model's weights
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=0)

  datagen = ImageDataGenerator(
    horizontal_flip=True,  # random horizontal flip
    width_shift_range=0.2,  # randomly shift images horizontally (20% of the width)
    height_shift_range=0.2,  # randomly shift images vertically (20% of the height)
    fill_mode='reflect',  # reflect padding mode
  )

  train_generator = datagen.flow(*ds, batch_size=128)

  model.fit(train_generator, validation_data=val_ds, epochs=TRAIN_EPOCHS, callbacks=[cp_callback, CustomCallback(), LearningRateScheduler(scheduler)], verbose=1)#, validation_data=val_ds)
  model.load_weights(checkpoint_path)
  return model


def predictImages(images, model):                    # (height, width, channels)
  predictions = model.predict(images)
  score = tf.nn.softmax(predictions)
  #print(score)
  return score

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor

def getPredictions(ds, model):
  val_ds_unbatched = list(ds.unbatch())
  labels = [tf.get_static_value(x[1]) for x in val_ds_unbatched]
  images = [x[0] for x in val_ds_unbatched]
  images = np.asarray(images)
  predictions = predictImages(images, model)
  return predictions, labels


def getAccuracy(ds, model):
  results = model.evaluate(ds, batch_size=BATCH_SIZE, verbose=0)
  results[1] *= 100
  results = results[::-1]
  return results

def reorder(chosenset):
  oldchosenset = chosenset
  chosenset = list(chosenset)
  chosenset.sort(key = lambda x: x.imagename)
  """for index in range(len(chosenset)):
    if chosenset[index]!=oldchosenset[index] and random.randint(0, 100) == 10: 
      print(chosenset[index].oldimagename)
      cv2_imshow(chosenset[index])"""
  return chosenset

def saveSets(set1, set2, val_images):
  saveSet(set1, "set1")
  saveSet(set2, "set2")
  saveSet(val_images, "val_images")


def load_datasets():
  with open(os.devnull, "w") as f:
    with redirect_stdout(f):
      set1_ds = tf.keras.utils.image_dataset_from_directory(
        "set1",
        seed=123,
        image_size=(HEIGHT, WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=False,
      )

      set2_ds = tf.keras.utils.image_dataset_from_directory(
        "set2",
        seed=123,
        image_size=(HEIGHT, WIDTH),
        shuffle=False,
      )

      val_ds = tf.keras.utils.image_dataset_from_directory(
        "val_images",
        seed=123,
        image_size=(HEIGHT, WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=False,
      )

  return set1_ds, set2_ds, val_ds

def modifySet(set2, thresh=2, thesh1=0.6):
  global dataset_modification_progress
  global accuracy_increase
  global accuracy_decrease

  incorrectChange = 0
  correctChange = 0

  print(type(set2))
  print(type(set2[0]))
  print(type(labels))
  print(type(labels[0]))

  for i in range(len(set2)):
    if i%(len(set2)//20) == 0:
      dataset_modification_progress = i
      loading_bar.display()
    #cv2_imshow(images[i])
    #cv2_imshow(set2[i].image)
    idealindex = labels.index(set2[i].imagename)
    scoreatindex = predictions[i][idealindex]
    scoreatindex = tf.get_static_value(scoreatindex)
    maxscore = np.max(predictions[i])
    if maxscore/thresh > scoreatindex and maxscore > thesh1:
      newlabel = labels[np.argmax(predictions[i])]
      if set2[i].imagename == set2[i].oldimagename:
        incorrectChange += 1
      elif set2[i].oldimagename == newlabel:
        correctChange += 1
      set2[i].imagename = newlabel
  
  accuracy_increase = (correctChange/len(set2))*100
  accuracy_decrease = (incorrectChange/len(set2))*100
  dataset_modification_progress = ((len(set2)//20)+1)*20
  loading_bar.display()
  set2 = reorder(set2)

  return set2

def deleteFromSet(set2, thresh=2, thesh1=0.6):
  global dataset_modification_progress
  global accuracy_increase
  global accuracy_decrease

  incorrectChange = 0
  correctChange = 0

  newset2 = []

  for i in range(len(set2)):
    if i%(len(set2)//20) == 0:
      dataset_modification_progress = i
      loading_bar.display()
    #cv2_imshow(images[i])
    #cv2_imshow(set2[i].image)
    idealindex = labels.index(set2[i].imagename)
    scoreatindex = predictions[i][idealindex]
    scoreatindex = tf.get_static_value(scoreatindex)
    maxscore = np.max(predictions[i])
    if maxscore/thresh > scoreatindex and maxscore > thesh1:
      if set2[i].imagename == set2[i].oldimagename:
        incorrectChange += 1
      else:
        correctChange += 1
    else:
      newset2.append(set2[i])
  
  set2 = newset2
  
  accuracy_increase = (correctChange/len(set2))*100
  accuracy_decrease = (incorrectChange/len(set2))*100
  dataset_modification_progress = ((len(set2)//20)+1)*20
  loading_bar.display()
  set2 = reorder(set2)

  return set2

def swapSets(set1, set2):
  temp = list(set2)
  set2 = list(set1)
  set1 = temp
  return set1, set2



def trainEpochs(images, val_images, epochs, verbose=1, mode="modify"):

  global epoch, model, predictions, truelabels, set1, set2, half, loading_bar, epochtime, train_epoch, set1_ds, set2_ds, val_ds, val_accuracy, val_loss, dataset_modification_progress, dataset_accuracy_before, dataset_accuracy_after, accuracy_increase, accuracy_decrease, total_accuracy

  loading_bar = LoadingBar(verbose)

  num_classes = 10


  x_train, y_train, y_train_old = images
  x_test, y_test = val_images

  val_accuracy_list = []
  val_loss_list = []
  val_accuracy_list = []
  dataset_accuracy_before_list = []
  dataset_accuracy_after_list = []
  accuracy_increase_list = []
  accuracy_decrease_list = []
  total_accuracy_list = []

  for epoch in range(epochs):
    set1, set2 = splitBuildingSet(x_train, y_train, y_train_old, 0.5)

    for half in range(2):
      epochtime = time.time()
      train_epoch = 0

      set1Encoded, val_imagesEncoded = convertToUseful(*set1[:2], x_test, y_test)
      set2Encoded, _ = convertToUseful(*set2[:2], x_test, y_test)
      
      val_accuracy = "?"
      val_loss = "?"
      dataset_modification_progress = 0
      dataset_accuracy_before = "?"
      dataset_accuracy_after = "?"
      accuracy_increase = "?"
      accuracy_decrease = "?"
      total_accuracy = "?"

      loading_bar.display()

      model = trainModel(set1Encoded, val_imagesEncoded)


      val_accuracy, val_loss = getAccuracy(val_ds, model)
      loading_bar.display()

      predictions, truelabels = getPredictions(set2_ds, model)

      dataset_accuracy_before = getLabelingAccuracy(set2)
      loading_bar.display()

      if mode == "modify":
        set2 = modifySet(set2, thresh=2)
      else:
        set2 = deleteFromSet(set2, thresh=2)
      

      dataset_accuracy_after = getLabelingAccuracy(set2)
      loading_bar.display()

      if half == 0:
        set1, set2 = swapSets(set1, set2)
      
      total_accuracy = getLabelingAccuracy(reorder(set1+set2))
      loading_bar.display(save=True)


      val_accuracy_list.append(val_accuracy)
      val_loss_list.append(val_loss)
      dataset_accuracy_before_list.append(dataset_accuracy_before)
      dataset_accuracy_after_list.append(dataset_accuracy_after)
      accuracy_increase_list.append(accuracy_increase)
      accuracy_decrease_list.append(accuracy_decrease)
      total_accuracy_list.append(total_accuracy)


    images = set1+set2
    images = reorder(images)

  metadata = {"val accuracy": val_accuracy_list, "val loss": val_loss_list, "dataset accuracy before": dataset_accuracy_before_list, "dataset accuracy after": dataset_accuracy_after_list, "dataset correct relabelling": accuracy_increase_list, "dataset incorrect relabelling": accuracy_decrease_list, "total dataset accuracy": total_accuracy_list}

  return model, images, metadata

def displayGraph(metadata):
  fig, ax = plt.subplots(figsize=(12, 8))


  for key in metadata:
    if key in ["val accuracy", "dataset accuracy before", "dataset accuracy after", "dataset correct relabelling", "dataset incorrect relabelling", "total dataset accuracy"]:
      plt.plot([i/2 for i in range(1, len(metadata[key])+1)], metadata[key], label=key.title())
  plt.yticks(np.arange(0, 100.01, 10))
  plt.legend()
  plt.grid()
  plt.title("Percentage values against half epochs")
  ax.set_ylabel("Percentage")
  ax.set_xlabel("Epochs")
  plt.show()

  fig, ax = plt.subplots(figsize=(12, 8))

  for key in metadata:
    if key in ["val loss"]:
      plt.plot([i/2 for i in range(1, len(metadata[key])+1)], metadata[key], label=key.title())
      plt.yticks(np.arange(0, max(metadata[key]), 0.1))
  plt.legend()
  plt.grid()
  plt.title("Loss against half epochs")
  ax.set_ylabel("Loss")
  ax.set_xlabel("Epochs")
  plt.show()

def getCorrectSample(images):
  correct = []
  incorrect = []
  for image in images:
    if image.imagename == image.oldimagename:
      correct.append(image)
    else:
      incorrect.append(image)
  return correct, incorrect