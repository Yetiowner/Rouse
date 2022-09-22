from mnist import MNIST
import random
import numpy as np
import cv2
import os
import shutil

mndata = MNIST('mnist')

images, labels = mndata.load_training()
# or
#images, labels = mndata.load_testing()

index = random.randrange(0, len(images))  # choose an index ;-)

newimages = []
for image in images:
    newimage = []
    size = int(len(image)**0.5)

    progress = size
    while progress <= size**2:
        newimage.append(image[progress-size:progress])
        progress += size
    newimages.append(newimage)

images = np.array(newimages)

for i in range(10):
    shutil.rmtree(f"dataset/{i}")
    os.mkdir(f"dataset/{i}")

for imageindex in range(len(images)):
    cv2.imwrite(f"dataset/{labels[imageindex]}/{imageindex}.jpg", images[imageindex])