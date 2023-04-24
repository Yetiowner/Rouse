import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras

def lq_loss(y_true, y_pred, q=0.4):
    Lq = (1 - tf.math.pow(y_pred, q)) / q
    loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(y_true, Lq), axis=1))
    return loss

def createModel(inputshape = (32, 32, 3), outputclasses = 10, decay = 0.0001, lr = 0.01, momentum = 0.9, lq=0.4):


  print("Q: ", lq)

  inputs = Input(shape=inputshape)
  x = BatchNormalization()(inputs)
  x = Activation('relu')(x)
  x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(decay))(x)

  # Residual blocks
  num_layers = [3, 4, 6, 3]
  filters = [64, 128, 256, 512]
  for i, n in enumerate(num_layers):
      for j in range(n):
          # downsampling in first layer of each block
          strides = (2, 2) if j == 0 and i > 0 else (1, 1)
          x_shortcut = x
          x = BatchNormalization()(x)
          x = Activation('relu')(x)
          x = tf.keras.layers.Conv2D(filters=filters[i], kernel_size=(3, 3), strides=strides, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(decay))(x)
          x = BatchNormalization()(x)
          x = Activation('relu')(x)
          x = tf.keras.layers.Conv2D(filters=filters[i], kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(decay))(x)
          # residual connection
          if strides != (1, 1) or x.shape[-1] != x_shortcut.shape[-1]:
              x_shortcut = tf.keras.layers.Conv2D(filters=filters[i], kernel_size=(1, 1), strides=strides, padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(decay))(x_shortcut)
          x = tf.keras.layers.Add()([x, x_shortcut])

  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  # Global average pooling and final fully connected layer
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  outputs = Dense(outputclasses, activation='softmax')(x)

  model = Model(inputs, outputs)

  # Learning rate schedule

  # Optimizer
  optimizer = SGD(learning_rate=lr, momentum=momentum, nesterov=True)

  # Compile the model
  model.compile(optimizer=optimizer, loss=(lambda y_true, y_pred: lq_loss(y_true, y_pred, lq)) if lq != 0 else 'categorical_crossentropy', metrics=['accuracy'])

  return model

def resnet_block(inputs, num_filters, downsample=False, decay = 0.0001):
    shortcut = inputs
    stride = 1
    if downsample:
        stride = 2
        shortcut = keras.layers.Conv2D(num_filters, 1, strides=stride, padding='same',
                                          kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(decay))(shortcut)
    inputs = keras.layers.Conv2D(num_filters, 3, strides=stride, padding='same',
                                    kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(decay))(inputs)
    inputs = keras.layers.BatchNormalization()(inputs)
    inputs = keras.layers.Activation('relu')(inputs)
    inputs = keras.layers.Conv2D(num_filters, 3, strides=1, padding='same',
                                    kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(decay))(inputs)
    inputs = keras.layers.BatchNormalization()(inputs)
    shortcut = keras.layers.BatchNormalization()(shortcut)
    inputs = keras.layers.Add()([inputs, shortcut])
    inputs = keras.layers.Activation('relu')(inputs)
    return inputs

def createFastModel(inputshape = (32, 32, 3), outputclasses = 10, decay = 0.0001, lr = 0.01, momentum = 0.9, lq=0.4):

  # Define ResNet34 architecture
  inputs = keras.layers.Input(shape=inputshape)
  x = keras.layers.Conv2D(64, 7, strides=2, padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(decay))(inputs)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.ReLU()(x)
  x = keras.layers.MaxPool2D(3, strides=2, padding='same')(x)

  # Residual blocks
  x = resnet_block(x, 64, decay=decay)
  x = resnet_block(x, 64, decay=decay)
  x = resnet_block(x, 64, decay=decay)

  x = resnet_block(x, 128, downsample=True, decay=decay)
  x = resnet_block(x, 128, decay=decay)
  x = resnet_block(x, 128, decay=decay)
  x = resnet_block(x, 128, decay=decay)

  x = resnet_block(x, 256, downsample=True, decay=decay)
  x = resnet_block(x, 256, decay=decay)
  x = resnet_block(x, 256, decay=decay)
  x = resnet_block(x, 256, decay=decay)
  x = resnet_block(x, 256, decay=decay)
  x = resnet_block(x, 256, decay=decay)

  x = resnet_block(x, 512, downsample=True, decay=decay)
  x = resnet_block(x, 512, decay=decay)
  x = resnet_block(x, 512, decay=decay)

  # Final layers
  x = keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.Flatten()(x)
  x = keras.layers.Dense(outputclasses, activation='softmax')(x)

  # Create model
  model = keras.models.Model(inputs=inputs, outputs=x)

  # Compile model
  opt = keras.optimizers.SGD(learning_rate=lr, momentum=momentum, nesterov=True)
  model.compile(optimizer=opt, loss=(lambda y_true, y_pred: lq_loss(y_true, y_pred, lq)) if lq != 0 else 'categorical_crossentropy', metrics=['accuracy'])
  model.summary()

  return model