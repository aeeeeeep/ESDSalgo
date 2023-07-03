import argparse
import gc
import numpy as np
import os
import shutil
# import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from collections import OrderedDict
from torch.autograd import Variable
from torchvision import datasets, transforms
from tqdm import tqdm

# Read image filenames from the dataset folders

data_dir = './data/plate'
class_names0 = os.listdir(data_dir)

class_names = []
for item in class_names0:
    class_names += [item]
num_class = len(class_names)
image_files = [[os.path.join(data_dir, class_name, x) \
                for x in os.listdir(os.path.join(data_dir, class_name))[:100000]] \
               for class_name in class_names]

image_file_list = []
image_label_list = []
for i, class_name in enumerate(class_names):
    image_file_list.extend(image_files[i])
    image_label_list.extend([i] * len(image_files[i]))
num_total = len(image_label_list)

width, height = Image.open(image_file_list[0]).size

print("Total image count:", num_total)
print("Image dimensions:", width, "x", height)
print("Label names:", class_names)
print("Label counts:", [len(image_files[i]) for i in range(num_class)])

# Prepare training, validation, and test data lists
valid_frac = 0.2
trainX, trainY = [], []
valX, valY = [], []
testX, testY = [], []

if not os.path.exists('./plate_train'):
    print("Create dataset...")
    for i in tqdm(range(num_total)):
        rann = np.random.random()
        if rann < valid_frac:
            valX.append(image_file_list[i])
            valY.append(image_label_list[i])
            j = image_file_list[i]
            k = j.split('/')[-2]
            r = j.split('/')[-1]
            os.makedirs(f'./plate_val/{k}', exist_ok=True)
            shutil.copyfile(j, os.path.join(f'./plate_val/{k}/', r))
        else:
            trainX.append(image_file_list[i])
            trainY.append(image_label_list[i])
            j = image_file_list[i]
            k = j.split('/')[-2]
            r = j.split('/')[-1]
            os.makedirs(f'./plate_train/{k}', exist_ok=True)
            shutil.copyfile(j, os.path.join(f'./plate_train/{k}/', r))
else:
    trainX = os.listdir('./plate_train')
    valX = os.listdir('./plate_val')

print(len(trainX), len(valX))

train_dir = './plate_train'
val_dir = './plate_val'
batch_size = 64

transform_train = transforms.Compose(
    [transforms.ToTensor(),
     transforms.RandomRotation(10, center=(0, 0), expand=True),
     transforms.RandomCrop(64, padding=8, padding_mode='edge'),
     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
     # transforms.ColorJitter(hue=0.3),
     transforms.RandomApply([transforms.RandomAffine(degrees=0, shear=(0, 0, 0, 30))], p=0.5),
     transforms.Resize((64, 64)), ])
# transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

transform_val = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((64, 64)), ])
# transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

image_datasets = {}
image_datasets["train"] = datasets.ImageFolder(root=train_dir, transform=transform_train)
image_datasets["valid"] = datasets.ImageFolder(root=val_dir, transform=transform_val)

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes

print(class_names)

train_loader = torch.utils.data.DataLoader(image_datasets["train"], batch_size=batch_size, shuffle=True,
                                           num_workers=8)
valid_loader = torch.utils.data.DataLoader(image_datasets["valid"], batch_size=batch_size, shuffle=False,
                                           num_workers=8)

print(dataset_sizes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

parser = argparse.ArgumentParser()

parser.add_argument('-save_dir', action="store", dest="save_dir", type=str, default="./work_dirs_plate")
parser.add_argument('-lr', action="store", dest="lr", type=float, default=1e-4)
parser.add_argument('-hiddenunits', action="store", dest="hiddenunits", type=int, default=128)
parser.add_argument('-epochs', action="store", dest="epochs", type=int, default=3)
ins = parser.parse_args(args=[])


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channel = 3
        self.out_channel = 32
        self.backbone = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.in_channel, self.out_channel, 3, 1, 1)),
            ('bn', nn.BatchNorm2d(self.out_channel)),
            ('relu', nn.ReLU(inplace=True)),
            ('maxpool', nn.MaxPool2d(3, stride=2, padding=1))
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ('dropout1', nn.Dropout(0.3)),
            ('fc1', nn.Linear(32768, ins.hiddenunits)),
            ('relu1', nn.ReLU(inplace=True)),
            ('dropout2', nn.Dropout(0.1)),
            ('fc2', nn.Linear(ins.hiddenunits, 2)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

    def forward(self, inputs):
        x = self.backbone(inputs)
        x = torch.flatten(x, 1)
        output = self.classifier(x)
        return output


model = Model()
print("Model:\n", model)

gc.collect()
torch.cuda.empty_cache()

use_gpu = torch.cuda.is_available()
print("Use GPU: ", use_gpu)

steps = int(len(trainX) / batch_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=ins.lr)

model.cuda()

train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []
best_acc = 0

print("Training...")
# Training
# with wandb.init(project="FWWB_A35", name="adam_1e-4", group="plate") as run:
for epoch in range(ins.epochs):
    # Reset variables at 0 epoch
    correct = 0
    iteration = 0
    iter_loss = 0.0

    model.train()  # Training Mode

    with tqdm(total=len(train_loader)) as _tqdm:
        _tqdm.set_description('epoch: {}/{}'.format(epoch, ins.epochs - 1))
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = Variable(inputs)
            labels = Variable(labels)
            CUDA = torch.cuda.is_available()
            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()  # clear gradient
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            iter_loss += loss.item()  # Accumulate loss
            loss.requires_grad_(True)
            loss.backward()  # backpropagation
            optimizer.step()  # update weights

            # Save the correct predictions for training data
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted.cpu() == labels.cpu()).sum()
            correct += accuracy
            iteration += 1

            # run.log({
            #     'loss': loss.item(),
            #     'accuracy': (accuracy / len(labels)),
            #     'lr': optimizer.state_dict()['param_groups'][0]['lr'],
            #     'epoch': epoch
            #     })
            _tqdm.set_postfix(loss='{:.3f}'.format(loss), accuracy='{:.3f}'.format(accuracy / len(labels)),
                              lr='{:.1E}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
            _tqdm.update(1)

    train_loss.append(iter_loss / iteration)
    train_accuracy.append((100 * correct / len(image_datasets["train"])))

    # Testing
    correct = 0
    iteration = 0
    valid_loss = 0.0
    print("Testing...")

    model.eval()  # Testing Mode

    with tqdm(total=len(valid_loader)) as _tqdm:
        for i, (inputs, labels) in enumerate(valid_loader):
            inputs = Variable(inputs)
            labels = Variable(labels)

            CUDA = torch.cuda.is_available()
            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted.cpu() == labels.cpu()).sum()

                iteration += 1

            _tqdm.update(1)

    test_loss.append(valid_loss / iteration)
    test_accuracy.append((100 * correct / len(image_datasets["valid"])))

    # run.log({
    #     'test_accuracy': test_accuracy[-1],
    #     'test_loss': test_loss[-1],
    #     'epoch': epoch
    #     })

    print('Epoch {}/{}, Training Loss:{:.3f}, Training Accuracy:{:.3f}, Testing Loss {:.3f}, Testing Accuracy:{:.3f}'
          .format(epoch + 1, ins.epochs, train_loss[-1], train_accuracy[-1], test_loss[-1], test_accuracy[-1]))
    # if (epoch + 1) % 4 == 0 :
    state_dict = model.module.state_dict() if next(model.parameters()).device == 'cuda:0' else model.state_dict()
    torch.save({'epoch': epoch, 'model_state_dict': state_dict},
               f'./{ins.save_dir}/model_epoch_{epoch + 1}.pth')
