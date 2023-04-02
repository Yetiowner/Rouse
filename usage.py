from Rouse import *
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


print("found", len(x_train), "images!")
showSample(x_test, y_test, 16)
y_train, y_train_old = swapLabels(y_train, 0.3)

print("Labeling accuracy at start:", getLabelingAccuracy(y_train, y_train_old))
#showSample(changes, 16)

model, images, metadata = trainEpochs((x_train, y_train, y_train_old), (x_test, y_test), MAIN_EPOCHS, verbose = 1)

displayGraph(metadata)

model1, images1, metadata1 = trainEpochs((x_train, y_train, y_train_old), (x_test, y_test), MAIN_EPOCHS, verbose = 1, mode="delete")

displayGraph(metadata1)