import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import skimage
from torch.utils.data import DataLoader, Dataset
import os.path
import seaborn as sns

from time import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TEST_DATA  = 'drive/My Drive/Data/Kaggle-MNIST/test.csv'
SAVE_FILE = 'model.nn'

# Disable all warnings (for when running on Colab)
import warnings
warnings.filterwarnings("ignore")


# Check for CUDA avilability
if(torch.cuda.is_available()):
    CUDA = True
    device = 'cuda:0'
else:
    CUDA = False
    device = 'cpu'
print('Device: ', device)




# Dataset for MNIST images
class DatasetMNIST(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    # Returns data and its label at some index
    def __getitem__(self, index):
        # Transform into (C, H, W) from (H, W, C)
        image = self.x[index,:].reshape((1, 28, 28))
        label = self.y[index]
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label



# Best model
class Net(nn.Module):
    def __init__(self, X, Y):
        super(Net, self).__init__()
        # Number of feature maps (filters) for convolutional layers 1 and 2
        fm1 = 32
        fm2 = 64

        self.conv1 = nn.Conv2d(1, fm1, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.0)
        self.batchnorm1 = nn.BatchNorm2d(fm1)
        self.conv2 = nn.Conv2d(fm1, fm2, 5)
        self.batchnorm2 = nn.BatchNorm2d(fm2)
        self.fc1 = nn.Linear(fm2 * 4 * 4, 128)
        self.batchnorm3 = nn.BatchNorm1d(self.fc1.out_features)
        self.fc2 = nn.Linear(128, 10)
        #self.fc1 = nn.Linear(16 * 4 * 4, X)
        #self.fc2 = nn.Linear(X, Y)
        #self.fc3 = nn.Linear(Y, 10)

    def forward(self, x):
        x = self.dropout(self.pool(F.relu(self.conv1(x))))
        #x = self.batchnorm1(x)
        x = self.dropout(self.pool(F.relu(self.conv2(x))))
        #x = self.batchnorm2(x)
        x = x.view(-1, self.fc1.in_features)
        x = self.dropout(F.relu(self.fc1(x)))
        #x = self.batchnorm3(x)
        x = F.softmax(self.fc2(x))
        return x








x_test = pd.io.parsers.read_csv(TEST_DATA).values / 255.0
x_test = x_test.astype(np.float32)
y_test = np.zeros(len(x_test), dtype='int64')
test_dataset = DatasetMNIST(x_test, y_test)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

print(x_test.shape, y_test.shape)


net = Net(0, 0)
if(CUDA):
  net.to(device)


# If a previous model exists, load it
if(os.path.isfile(SAVE_FILE)):
    print('Previous model "%s" found, loading...' % SAVE_FILE, end='')
    net = torch.load(SAVE_FILE)
    print('loaded.')



count = {}
for i in range(10):
    count[i] = 0

  
f = open('submission.csv','w')
f.write('ImageId,Label\n')

with torch.no_grad():
    i = 0
    for data in testloader:
        i += 1
        
        if(CUDA):
            images, labels = data[0].to(device), data[1].to(device)
        else:
            images, labels = data

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        
        f.write(str(i) + ',' + str(predicted.item()) + '\n')
        #print(predicted.item())
        count[predicted.item()] += 1
        
        if(i % 1000 == 0):
          print(i)
f.close()
print(count)







