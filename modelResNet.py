import keras

def resnet_block(inputs, num_filters, downsample=False, decay = 0.001):
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

def createModel(inputshape = (32, 32, 3), outputclasses = 10, lr = 0.0001, momentum = 0.9, decay = 0.001):

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
  x = keras.layers.Dense(outputclasses, activation='softmax')(x)

  # Create model
  model = keras.models.Model(inputs=inputs, outputs=x)

  # Compile model
  opt = keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
  
  return model