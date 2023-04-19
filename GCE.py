'''
ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


import torchvision
import torchvision.transforms as transforms

import os
import argparse
import csv
import time
from PIL import Image
import sys


last_time = time.time()
begin_time = last_time
term_width = 200
TOTAL_BAR_LENGTH = 65.

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)

def ResNet34(num_classes):
    return ResNet(BasicBlock, [3,4,6,3], num_classes)

class TruncatedLoss(nn.Module):

    def __init__(self, q=0.7, k=0.5, trainset_size=50000):
        super(TruncatedLoss, self).__init__()
        self.q = q
        self.k = k
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, 1), requires_grad=False)
             
    def forward(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets.long(), 1))

        loss = ((1-(Yg**self.q))/self.q)*self.weight[indexes] - ((1-(self.k**self.q))/self.q)*self.weight[indexes]
        loss = torch.mean(loss)

        return loss

    def update_weight(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        Lq = ((1-(Yg**self.q))/self.q)
        Lqk = np.repeat(((1-(self.k**self.q))/self.q), targets.size(0))
        Lqk = torch.from_numpy(Lqk).type(torch.cuda.FloatTensor)
        Lqk = torch.unsqueeze(Lqk, 1)
        

        condition = torch.gt(Lqk, Lq)
        self.weight[indexes] = condition.type(torch.cuda.FloatTensor)



class MyDataset(Dataset):
    def __init__(self, x_train, y_train, transform=None):
        self.x_train = x_train
        self.y_train = y_train
        self.transform = transform
        self.target_transform = None
        
    def __len__(self):
        return len(self.x_train)
    
    def __getitem__(self, index):
        img, target = self.x_train[index], self.y_train[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

class PytorchWithTensorflowCapabilities:
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion
    
    def evaluate(self, x_train, y_train, batch_size=128, verbose=0):
      y_train = torch.from_numpy(y_train)
      print(y_train)
      print(y_train.shape)

      predicted = torch.from_numpy(self.predict(x_train))
      actual = torch.eye(10)[y_train]

      loss = nn.CrossEntropyLoss()(predicted, actual).item()
      true_labels = torch.tensor(true_labels)

      # Get the predicted class labels by taking the argmax along the second axis (axis=1)
      pred_class_labels = torch.argmax(predicted, dim=1)

      # Compare pred_class_labels with true_labels to get a boolean tensor of shape (50000,)
      correct_preds = (pred_class_labels == y_train)

      # Calculate the accuracy as the mean of correct_preds
      accuracy = torch.mean(correct_preds.float())

      # Convert accuracy to a scalar float value
      accuracy = accuracy.item()

      return loss, accuracy

    def predict(self, images, softmax = True):
      
      transform_images = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
      ])

      transformed_images = []
      for i in range(len(images)):
          transformed_image = transform_images(images[i])
          transformed_images.append(transformed_image)

      batch_tensor = torch.stack(transformed_images, dim=0)

      self.model.eval()

      # Disable gradient computation
      with torch.no_grad():
        outputs = self.model(batch_tensor)
      
      if softmax:
        outputs = torch.softmax(outputs, dim=1)
      
      return outputs.cpu().numpy()

best_acc = 0

def train_main(train_images, val_images, q = 0.7, epochcount = 50, num_classes = 10, lr = 0.01, decay = 1e-4, start_prune = 40, epoch_callbacks = []) -> PytorchWithTensorflowCapabilities:
    
    use_cuda = torch.cuda.is_available()
    global best_acc 

    x_train, y_train = train_images
    x_test, y_test = val_images
 
    # load dataset
        
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
    ])

    train_dataset = MyDataset(x_train, y_train, transform_train)
    
    test_dataset = MyDataset(x_test, y_test, transform_test)

    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=2)

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=2)
    # Model
    """if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.t7.' + args.sess)
        net = checkpoint['net']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        torch.set_rng_state(checkpoint['rng_state'])
    else:"""

    #print('==> Building model.. (Default : ResNet34)')
    start_epoch = 0
    net = ResNet34(num_classes)

    result_folder = './results/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    logname = result_folder + net.__class__.__name__ + \
        '_' + "default" + '.csv'

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net)
        #print('Using', torch.cuda.device_count(), 'GPUs.')
        cudnn.benchmark = True
        #print('Using CUDA..')


    criterion = TruncatedLoss(trainset_size=len(train_dataset), q = q).cuda()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[40, 80], gamma=0.1)

    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(
                ['epoch', 'train loss', 'train acc', 'test loss', 'test acc'])

    for epoch in range(start_epoch, epochcount):
        
        train_loss, train_acc = train(epoch, start_prune, trainloader, net, criterion, optimizer)
        test_loss, test_acc = test(epoch, testloader, net, criterion)

        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, train_acc, test_loss, test_acc])
        scheduler.step()

        for epoch_callback in epoch_callbacks:
          epoch_callback(epoch)
  
    return PytorchWithTensorflowCapabilities(net, criterion)

# Training
def train(epoch, start_prune, trainloader, net, criterion, optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    if (epoch+1) >= start_prune and (epoch+1) % 10 == 0:
        checkpoint = torch.load('./checkpoint/ckpt.t7.' + "default")
        net = checkpoint['net']
        net.eval()
        for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            criterion.update_weight(outputs, targets, indexes)
        now = torch.load('./checkpoint/current_net')
        net = now['current_net']
        net.train()
    
    for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = net(inputs)
        loss = criterion(outputs, targets, indexes)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
             
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        correct = correct.item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


    return (train_loss / batch_idx, 100. * correct / total)


def test(epoch, testloader, net, criterion):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets, indexes)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct = correct.item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))


    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        checkpoint(acc, epoch, net)
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')

    state = {
        'current_net': net,
    }
    torch.save(state, './checkpoint/current_net')
    return (test_loss / batch_idx, 100. * correct / total)

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    """for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')"""
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def checkpoint(acc, epoch, net):
    # Save checkpoint.
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.t7.' +
               "default")