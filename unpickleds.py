import os
import tarfile
import numpy as np
from PIL import Image

# Extract the CIFAR-10 dataset file
tar = tarfile.open('cifar-10-python.tar.gz', 'r:gz')
tar.extractall()
tar.close()

# Load the data batch files
data_batches = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Create the output directory
output_dir = 'dataset'
os.makedirs(output_dir, exist_ok=True)

for label_name in label_names:
  os.makedirs(os.path.join(output_dir, label_name), exist_ok=True)

# Save the images as JPEG files
startval = 0
for data_batch in data_batches:
    print(data_batch)
    with open("cifar-10-batches-py" + "\\" + data_batch, 'rb') as f:
        data_dict = np.load(f, encoding='bytes', allow_pickle=True)
    data = data_dict[b'data']
    labels = data_dict[b'labels']
    for i in range(len(data)):
        img = Image.fromarray(np.transpose(np.reshape(data[i], (3, 32, 32)), (1, 2, 0)))
        label_name = label_names[labels[i]]
        img.save(os.path.join(output_dir, label_name, '{}.jpg'.format(i+startval)))
    startval += i