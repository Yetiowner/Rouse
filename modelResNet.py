import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def createModel(inputshape = (32, 32, 3), outputclasses = 10, decay = 0.001, lr = 0.01, momentum = 0.9):

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
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

  return model