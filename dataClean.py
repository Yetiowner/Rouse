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
from collections import Counter

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from keras import models
from keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import categorical_crossentropy
from keras.preprocessing import image
from Rouse.GCE import train_main, PytorchWithTensorflowCapabilities
try:
  from google.colab.patches import cv2_imshow
except:
  pass

HEIGHT = 32
WIDTH = 32
CHANNELS = 3
BATCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 100
TRAIN_EPOCHS = 41
SECONDARY_EPOCHS = 5
MAIN_EPOCHS = 4
NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

"""if "display" not in globals():
  def display(*args, **kwargs):
    if len(args) == 0:
      return
    print(args[0])"""
if "cv2_imshow" not in globals():
  def cv2_imshow(img):
    cv2.imshow("window", img)

class RankPruningCallback(Callback):
    def __init__(self, x_train, y_train, train_generator, prune_every=10, prune_start=39, prune_ratio=0.1):
        super(RankPruningCallback, self).__init__()
        self.train_generator = train_generator
        self.x_train = x_train
        self.y_train = y_train
        self.prune_every = prune_every
        self.prune_start = prune_start
        self.prune_ratio = prune_ratio
        
    def on_epoch_end(self, epoch, logs=None):
        #global y_train_old
        
        if epoch >= self.prune_start and (epoch - self.prune_start) % self.prune_every == 0:
            # Rank the training examples based on classification confidence

            num_to_prune = int(self.prune_ratio * len(self.x_train)) # prune the bottom X%
            mode = "loss"

            if mode == "loss":
              y_pred = self.model.predict(self.x_train)
              y_true = self.y_train
              losses = categorical_crossentropy(self.y_train, y_pred).numpy()
              indices_to_prune = np.argsort(losses)[-num_to_prune:]

            else:
              confidence_scores = self.model.predict(self.x_train, verbose=False, batch_size=128)
              correct_class_scores = tf.gather_nd(confidence_scores, tf.stack([tf.range(self.y_train.shape[0]), tf.cast(tf.argmax(self.y_train, axis=1), tf.int32)], axis=1))
              max_class_scores = tf.reduce_max(confidence_scores, axis=1)
              confidence_ratios = correct_class_scores / max_class_scores
              ranks = tf.argsort(confidence_ratios, direction='ASCENDING')
              num_to_prune = min(num_to_prune, tf.math.count_nonzero(tf.math.not_equal(confidence_ratios, 1)).numpy()-1)
              indices_to_prune = ranks[:num_to_prune]



            #showSample(self.x_train[indices_to_prune], self.y_train[indices_to_prune], 16)


            self.train_generator.x = np.delete(self.train_generator.x, indices_to_prune, axis=0)
            self.train_generator.y = np.delete(self.train_generator.y, indices_to_prune, axis=0)

            self.x_train = np.delete(self.x_train, indices_to_prune, axis=0)
            self.y_train = np.delete(self.y_train, indices_to_prune, axis=0)

            """y_train_non_categorical = np.argmax(copy.deepcopy(self.y_train), axis=1).reshape((-1, 1))
            y_train_old = np.delete(y_train_old, indices_to_prune, axis=0)
            print("foo")
            print(getLabelingAccuracy(y_train_old, y_train_non_categorical))
            print("bar")"""

            self.train_generator.n -= num_to_prune
            self.train_generator._set_index_array()
            self.train_generator.reset()
            
            # Update the logs
            logs = logs or {}
            logs['pruning_ratio'] = self.prune_ratio
            logs['num_examples_pruned'] = num_to_prune

class CustomCallback(keras.callbacks.Callback):
    
    """def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))"""

    def on_epoch_begin(self, epoch, logs=None):
        global train_epoch
        train_epoch = epoch + 1
        loading_bar.display()

class LoadingBar():
  def __init__(self, verbose, main_epochs, train_epochs):
    self.structure = "| Elapsed: {} | Epoch {} | {} half epoch | Model training progress: {} {}/{} | Validation accuracy: {}% | Validation loss: {} | Dataset modification progress: {} {}/{} | Dataset half accuracy before half epoch: {}% | Dataset half accuracy after half epoch: {}% | Correct relabelling: {}% | Incorrect relabelling: {}% | Neutral relabelling: {}% | Total dataset accuracy: {}% |"
    self.out = display(display_id=True)
    self.verbose = verbose
    self.main_epochs = main_epochs
    self.train_epochs = train_epochs
  
  def display(self, save=False):

    if not(self.verbose):
      return

    time_elapsed = str(datetime.timedelta(seconds=int(time.time()-epochtime)))

    current_epoch = str((epoch + 1)).rjust(len(str(self.main_epochs)))
    half_epoch = " First" if half == 0 else "Second"

    progress_dash_train = "=" * train_epoch + "." * (self.train_epochs - train_epoch)
    epoch_train = str(train_epoch).rjust(len(str(self.train_epochs)))
    epoch_train_total = self.train_epochs

    val_accuracy_str = str(format(val_accuracy, ".2f") if val_accuracy != "?" else "?").rjust(5)
    val_loss_str = str(format(val_loss, ".3f") if val_loss != "?" else "?").rjust(5)

    dataset_modification_progress_dash_train = "=" * (dataset_modification_progress//(len(set2[0])//20)) + "." * (20 - (dataset_modification_progress//(len(set2[0])//20)))
    dataset_modification_progress_str = str(min(dataset_modification_progress, len(set2[0]))).rjust(len(str(len(set2[0]))))
    dataset_total_modifications_str = str(len(set2[0]))

    dataset_accuracy_before_str = str(format(dataset_accuracy_before, ".2f") if dataset_accuracy_before != "?" else "?").rjust(5)
    dataset_accuracy_after_str = str(format(dataset_accuracy_after, ".2f") if dataset_accuracy_after != "?" else "?").rjust(5)
    dataset_accuracy_increase = str(format(accuracy_increase, ".2f") if accuracy_increase != "?" else "?").rjust(5)
    dataset_accuracy_decrease = str(format(accuracy_decrease, ".2f") if accuracy_decrease != "?" else "?").rjust(5)
    dataset_accuracy_not_changed = str(format(accuracy_not_changed, ".2f") if accuracy_not_changed != "?" else "?").rjust(5)

    total_accuracy_str = str(format(total_accuracy, ".2f") if total_accuracy != "?" else "?").rjust(5)

    filledstructure = self.structure.format(time_elapsed, current_epoch, half_epoch, progress_dash_train, epoch_train, epoch_train_total, val_accuracy_str, val_loss_str, dataset_modification_progress_dash_train, dataset_modification_progress_str, dataset_total_modifications_str, dataset_accuracy_before_str, dataset_accuracy_after_str, dataset_accuracy_increase, dataset_accuracy_decrease, dataset_accuracy_not_changed, total_accuracy_str)
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

def loadingBarEpochCallback(epoch):
  global train_Epoch
  train_epoch = epoch + 1
  loading_bar.display()

def scheduler(epoch, lr):
    if epoch == 40 or epoch == 80:
        return lr * 0.1
    else:
        return lr

# ResNet34 architecture
def convertToUseful(x_train, y_train, x_test, y_test):
  y_train = y_train.ravel()
  y_test = y_test.ravel()

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

def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

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

def trainModel(ds, val_ds, epochcount = None, loadingBar = True, fast = True, q = None):
  model = train_main(ds, val_ds, q = q, epochcount = epochcount, num_classes = 10, lr = 0.01, decay = 0.0001)
  return model
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

  if fast:
    createModel = modelResNet.createModel
  else:
    createModel = modelResNet.createModel
  
  model = createModel(inputshape=(HEIGHT, WIDTH, CHANNELS), outputclasses=num_classes, lq=(q if q != None else (0.4 if fast else 0.4)))

  checkpoint_path = "training_1/cp.ckpt"
  checkpoint_dir = os.path.dirname(checkpoint_path)

  # Create a callback that saves the model's weights
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=0)

  datagen = ImageDataGenerator(
    horizontal_flip=True,  # random horizontal flip
    width_shift_range=4,  # randomly shift images horizontally (20% of the width)
    height_shift_range=4,  # randomly shift images vertically (20% of the height)
    fill_mode='reflect',  # reflect padding mode
  )

  train_generator = datagen.flow(*ds, batch_size=(128 if not fast else 64))

  callbacks = [cp_callback, LearningRateScheduler(scheduler), RankPruningCallback(*ds, train_generator, prune_ratio = (0.1 if fast else 0.1), prune_start=(39 if fast else 39), prune_every=(10 if fast else 10))]
  if loadingBar:
    callbacks.append(CustomCallback())

  model.fit(train_generator, validation_data=val_ds, epochs=epochcount, callbacks=callbacks, verbose=1)#, validation_data=val_ds)
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

def apply_data_augmentation(images, datagen, batch_size=64):
    """
    Applies data augmentation on a numpy array of images using an ImageDataGenerator with a specified batch size.

    Args:
        images (numpy array): Input numpy array of images with shape (num_samples, height, width, channels).
        datagen (ImageDataGenerator): Instance of ImageDataGenerator with desired data augmentation settings.
        batch_size (int): Batch size for generating augmented images. Default is 64.

    Returns:
        numpy array: Processed images with the same shape as the original images.
    """
    # Create an iterator to generate batches of images with the specified batch size
    image_iterator = datagen.flow(images, batch_size=batch_size, shuffle=False)

    # Initialize an empty numpy array to store the augmented images
    augmented_images = np.empty_like(images)

    # Iterate over the image_iterator to get batches of augmented images
    for i in range(len(images) // batch_size + 1):
        # Get the batch of augmented images
        batch = image_iterator.next()

        # Extract the augmented images from the batch and store them in the empty array
        augmented_images[i * batch_size:(i + 1) * batch_size] = batch

    return augmented_images

def getPredictions(ds, model, augmentationForModification):
  if augmentationForModification != -1:
    images = ds[0]
    datagen = ImageDataGenerator(
      rotation_range=15, # rotate images randomly by 15 degrees
      width_shift_range=0.3, # shift images horizontally by 10% of total width
      height_shift_range=0.3, # shift images vertically by 10% of total height
      shear_range=0.2, # apply shear transformation with intensity of 10%
      zoom_range=0.2, # zoom images randomly by 10%
      horizontal_flip=True, # flip images horizontally
      fill_mode='reflect', # fill mode for padding, uses reflection
      brightness_range=(0.9, 1.1), # randomly adjust brightness between 0.9 and 1.1
  )

    n = augmentationForModification

    # Create a list to store the versions of images
    augmented_images = []

    # Generate n versions of the images array
    print(1)
    for i in range(n):
        # Generate random variations of the images using datagen.flow()
        batch = apply_data_augmentation(images, datagen)
        #batch = datagen.flow(images, batch_size=len(images), shuffle=False).next()
        # Add the generated variations to the list
        augmented_images.append(batch)

    print(2)
    """for i in range(50):
      index = random.randint(0, len(images)-1)
      for j in range(augmentationForModification):
        cv2_imshow(augmented_images[j][index]*255)"""
    
    predictions = []
    for i in augmented_images:
      predictions.append(model.predict(i))
    
    print(3)

  else:
    predictions = model.predict(ds[0])
  return predictions


def getAccuracy(ds, model: PytorchWithTensorflowCapabilities):
  results = model.evaluate(*ds, batch_size=BATCH_SIZE, verbose=0)
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

def modifySet(set2, predictions, truelabels, augmentationForModification, thresh=3, thesh1=0.6, mu=0.2):
  global dataset_modification_progress
  global accuracy_increase
  global accuracy_decrease
  global accuracy_not_changed

  incorrectChange = 0
  correctChange = 0
  neutralChange = 0

  wrongChanges = []

  for i in range(len(set2[0])):
    if i%(len(set2[0])//20) == 0:
      print(i)
      dataset_modification_progress = i
      loading_bar.display()
    #cv2_imshow(images[i])
    #cv2_imshow(set2[i].image)

    if augmentationForModification == -1:
      idealindex = set2[1][i]
      scoreatindex = predictions[i][idealindex]
      scoreatindex = tf.get_static_value(scoreatindex)
      maxscore = np.max(predictions[i])
      newlabel = np.argmax(predictions[i])
      removalCondition = maxscore/thresh > scoreatindex and maxscore > thesh1
    else:

      maxlabels = []
      for predictionset in predictions:
        maxlabels.append(np.argmax(predictionset[i]))
      
      newlabel = most_common(maxlabels)

      removalCondition = (maxlabels.count(newlabel) > augmentationForModification*mu) and newlabel != set2[1][i]
      #removalCondition = (set2[1][i] != newlabel)

    if removalCondition:
      if set2[1][i] == truelabels[i][0]:
        incorrectChange += 1
        #wrongChanges.append([set2[0][i], set2[1][i], predictions[i], truelabels[i]])
      elif truelabels[i][0] == newlabel:
        correctChange += 1
      elif set2[1][i] != truelabels[i][0] and newlabel != truelabels[i][0]:
        neutralChange += 1
        #wrongChanges.append([set2[0][i], set2[1][i], predictions[i], truelabels[i]])
      set2[1][i] = newlabel
  
  """wrongImages = random.sample(wrongChanges, 50)
  for imageset in wrongImages:
    print("-----------------------")
    cv2_imshow(imageset[0])
    print("Original noisy label:")
    print(imageset[1])
    print("Machine learning prediction:")
    print(imageset[2])
    print("Original true label:")
    print(imageset[3])"""
  
  accuracy_increase = (correctChange/len(set2[0]))*100
  accuracy_decrease = (incorrectChange/len(set2[0]))*100
  accuracy_not_changed = (neutralChange/len(set2[0]))*100
  dataset_modification_progress = ((len(set2[0])//20)+1)*20
  loading_bar.display()

  return set2

def deleteFromSet(set2, predictions, truelabels, augmentationForModification, thresh=2, thesh1=0.6, mu = 0.2):
  global dataset_modification_progress
  global accuracy_increase
  global accuracy_decrease

  incorrectChange = 0
  correctChange = 0

  newset2 = [[], [], []]

  for i in range(len(set2[0])):
    if i%(len(set2[0])//20) == 0:
      print(i)
      dataset_modification_progress = i
      loading_bar.display()
    #cv2_imshow(images[i])
    #cv2_imshow(set2[i].image)

    if augmentationForModification == -1:
      idealindex = set2[1][i]
      scoreatindex = predictions[i][idealindex]
      scoreatindex = tf.get_static_value(scoreatindex)
      maxscore = np.max(predictions[i])
      removalCondition = maxscore/thresh > scoreatindex and maxscore > thesh1
    else:
      conditionsMetCount = 0

      for predictionset in predictions:
        idealindex = set2[1][i]
        scoreatindex = predictionset[i][idealindex]
        scoreatindex = tf.get_static_value(scoreatindex)
        maxscore = np.max(predictionset[i])
        if maxscore/thresh > scoreatindex and maxscore > thesh1:
          conditionsMetCount += 1
      
      removalCondition = conditionsMetCount > augmentationForModification*mu

    if removalCondition:
      if set2[1][i] == truelabels[i][0]:
        incorrectChange += 1
      else:
        correctChange += 1
    else:
      newset2[0].append(set2[0][i])
      newset2[1].append(set2[1][i])
      newset2[2].append(set2[2][i])
  
  set2 = tuple([np.array(i) for i in newset2])

  accuracy_increase = (correctChange/len(set2[0]))*100
  accuracy_decrease = (incorrectChange/len(set2[0]))*100
  dataset_modification_progress = ((len(set2[0])//20)+1)*20
  loading_bar.display()

  return set2

def swapSets(set1, set2):
  set1, set2 = set2, set1
  return set1, set2

def getValAccuracy(x_train, y_train, x_test, y_test, q = 0.4, epochs = 120):
  train_imagesEncoded, val_imagesEncoded = convertToUseful(x_train, y_train, x_test, y_test)
  model = trainModel(train_imagesEncoded, val_imagesEncoded, epochcount=epochs, loadingBar=False, fast=False, q = q)
  val_accuracy, val_loss = getAccuracy(val_imagesEncoded, model)
  return val_accuracy, val_loss

def trainEpochs(images, val_images, epochs, verbose=1, mode="modify", augmentationForModification=-1, saveOtherPartBeforeChanges = True, subEpochs = 41, subQValue = 0.4, monoepoch = False, mu = 0.7):

  global epoch, model, predictions, truelabels, set1, set2, half, loading_bar, epochtime, train_epoch, set1_ds, set2_ds, val_ds, val_accuracy, val_loss, dataset_modification_progress, dataset_accuracy_before, dataset_accuracy_after, accuracy_increase, accuracy_decrease, accuracy_not_changed, total_accuracy

  loading_bar = LoadingBar(verbose, main_epochs=epochs, train_epochs=subEpochs)

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
  accuracy_not_changed_list = []

  showNoiseMatrix(y_train, y_train_old, title="Noise distribution matrix on total dataset at the start")

  for epoch in range(epochs):
    if not monoepoch:
      set1, set2 = splitBuildingSet(x_train, y_train, y_train_old, 0.5)
    else:
      ds = [x_train, y_train, y_train_old]
      set1 = copy.deepcopy(ds)
      set2 = copy.deepcopy(ds)

    settotrainon = set1

    for half in range(2):
      epochtime = time.time()
      train_epoch = 0

      set1Encoded, val_imagesEncoded = convertToUseful(*settotrainon[:2], x_test, y_test)
      set2Encoded, _ = convertToUseful(*set2[:2], x_test, y_test)
      
      val_accuracy = "?"
      val_loss = "?"
      dataset_modification_progress = 0
      dataset_accuracy_before = "?"
      dataset_accuracy_after = "?"
      accuracy_increase = "?"
      accuracy_decrease = "?"
      accuracy_not_changed = "?"
      total_accuracy = "?"

      loading_bar.display()

      truelabels = set2[2]

      dataset_accuracy_before = getLabelingAccuracy(set2[1], truelabels)
      if verbose:
        showNoiseMatrix(set2[1], truelabels, title="Noise distribution matrix before modification")

      loading_bar.display()

      model = trainModel(set1Encoded, val_imagesEncoded, q=subQValue, epochcount=subEpochs)


      val_accuracy, val_loss = getAccuracy(val_imagesEncoded, model)
      loading_bar.display()

      predictions = getPredictions(set2Encoded, model, augmentationForModification)

      oldset2 = copy.deepcopy(set2)

      if saveOtherPartBeforeChanges:
        settotrainon = copy.deepcopy(set2) # This means that biases aren't fed forward

      if mode == "modify":
        set2 = modifySet(set2, predictions, truelabels, augmentationForModification, mu)
      else:
        set2 = deleteFromSet(set2, predictions, truelabels, augmentationForModification, mu)
        truelabels = set2[2]
      
      if not saveOtherPartBeforeChanges:
        settotrainon = copy.deepcopy(set2) # This means that biases are fed forward

      dataset_accuracy_after = getLabelingAccuracy(set2[1], truelabels)
      if verbose:
        showNoiseMatrix(set2[1], truelabels, title="Noise distribution matrix after modification at the end")
        showNoiseDifferenceMatrix(oldset2[1], set2[1], oldset2[2], truelabels)

      loading_bar.display()

      if half == 0:
        set1, set2 = set2, set1
      
      if monoepoch:
        total_accuracy = getLabelingAccuracy(set1[1], set1[2])
        loading_bar.display(save=True)
      else:
        total_accuracy = getLabelingAccuracy(np.concatenate([set1[1], set2[1]], axis=0), np.concatenate([set1[2], set2[2]], axis=0))
        loading_bar.display(save=True)


      val_accuracy_list.append(val_accuracy)
      val_loss_list.append(val_loss)
      dataset_accuracy_before_list.append(dataset_accuracy_before)
      dataset_accuracy_after_list.append(dataset_accuracy_after)
      accuracy_increase_list.append(accuracy_increase)
      accuracy_decrease_list.append(accuracy_decrease)
      accuracy_not_changed_list.append(accuracy_not_changed)
      total_accuracy_list.append(total_accuracy)

      if monoepoch:
        break


    if monoepoch:
      x_train = set1[0]
      y_train = set1[1]
      y_train_old = set1[2]
    else:
      x_train = np.concatenate([set1[0], set2[0]], axis=0)
      y_train = np.concatenate([set1[1], set2[1]], axis=0)
      y_train_old = np.concatenate([set1[2], set2[2]], axis=0)
  
  showNoiseMatrix(y_train, y_train_old, title="Noise distribution matrix on total dataset")

  metadata = {"val accuracy": val_accuracy_list, "val loss": val_loss_list, "dataset accuracy before": dataset_accuracy_before_list, "dataset accuracy after": dataset_accuracy_after_list, "dataset correct relabelling": accuracy_increase_list, "dataset incorrect relabelling": accuracy_decrease_list, "dataset neutral relabelling": accuracy_not_changed_list, "total dataset accuracy": total_accuracy_list}

  return model, (x_train, y_train, y_train_old), metadata

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

def showNoiseMatrix(noisy_labels, true_labels, names = NAMES, title = "Noise Distribution Matrix"):

# Compute the histogram
  hist, x_edges, y_edges = np.histogram2d(noisy_labels.flatten(), true_labels.flatten(), bins=10)
  hist = np.log2(hist + 0.001)

  # Create a new figure and axis
  fig, ax = plt.subplots()

  # Plot the histogram as a heatmap
  im = ax.imshow(hist, cmap='viridis')

  # Add a colorbar
  cbar = ax.figure.colorbar(im, ax=ax, format=lambda x, pos: f'{int(2**x)}')
  cbar.set_label("Relative frequency in log2 scale")

  # Set the tick labels for the x-axis
  ax.set_xticks(np.arange(len(names)))
  ax.set_xticklabels(names)

  # Set the tick labels for the y-axis
  ax.set_yticks(np.arange(len(names)))
  ax.set_yticklabels(names)

  # Rotate the tick labels for the x-axis
  plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
          rotation_mode="anchor")

  # Set the title and axis labels
  ax.set_title(title)
  ax.set_xlabel("True Labels")
  ax.set_ylabel("Noisy Labels")

  # Show the plot
  plt.show()

def showNoiseDifferenceMatrix(noisy_labels, new_noisy_labels, true_labels, new_true_labels, names = NAMES, title = "Noise difference matrix"):

# Compute the histogram
  hist, x_edges, y_edges = np.histogram2d(noisy_labels.flatten(), true_labels.flatten(), bins=10)
  hist1, x_edges, y_edges = np.histogram2d(new_noisy_labels.flatten(), new_true_labels.flatten(), bins=10)
  hist = hist1-hist

  # Create a new figure and axis
  fig, ax = plt.subplots()

  # Plot the histogram as a heatmap
  im = ax.imshow(hist, cmap='viridis')

  # Add a colorbar
  cbar = ax.figure.colorbar(im, ax=ax)
  cbar.set_label("Change in frequency")

  # Set the tick labels for the x-axis
  ax.set_xticks(np.arange(len(names)))
  ax.set_xticklabels(names)

  # Set the tick labels for the y-axis
  ax.set_yticks(np.arange(len(names)))
  ax.set_yticklabels(names)

  # Rotate the tick labels for the x-axis
  plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
          rotation_mode="anchor")

  # Set the title and axis labels
  ax.set_title(title)
  ax.set_xlabel("True Labels")
  ax.set_ylabel("Noisy Labels")

  # Show the plot
  plt.show()

def showAugmentedImages(image: np.array, n, datagen: ImageDataGenerator):
  """ Shows image in a vertical grid of n rows and 1 column"""
  image = image.reshape((1,) + image.shape)
  for i in range(n):
    cv2_imshow(datagen.flow(image, batch_size=1, shuffle=False).next()[0])