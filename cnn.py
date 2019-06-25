# Convolutional neural network


from util import *
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


# Disable all warnings (for when running on Colab)
import warnings
warnings.filterwarnings("ignore")


SAVE_FILE = 'model.nn'  # Filepath to save model to
BACKUP_INTERVAL = 2000  # Number of mini-batches between model backups
BATCH_SIZE = 4          # Size of a mini-batch
UPDATE_INTERVAL = 2000  # Number of mini-batches between loss updates
NUM_EPOCHS = 35         # Default number of epochs to train for
HYPERPARAM_TESTS = 1    # Number of random hyperparameter combinations to test

sns.set() # Initialize seaborn


# Check for CUDA avilability
if(torch.cuda.is_available()):
    CUDA = True
    device = 'cuda:0'
else:
    CUDA = False
    device = 'cpu'
print('Device: ', device)


# Show a given image
def imgshow(img):
    plt.imshow(img)
    plt.show()


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


# Load training and validation datasets
raw_data = load_csv(TRAIN_DATA)
x_train, x_val, y_train, y_val = split_data(raw_data, 1234)
x_train = x_train.astype(np.float32)
x_val = x_val.astype(np.float32)
x_train = x_train / 255.0
x_val = x_val / 255.0
# Augment training data
img_train = x_train.reshape((-1, 28, 28))
x_augments = []
y_augments = []
print('Augmenting data...')
for i, image in enumerate(img_train):
    rand_degree = np.random.uniform(-10, 10)
    new_img = skimage.transform.rotate(image, rand_degree)
    new_img = skimage.util.random_noise(new_image, seed=1234)
    x_augments.append(new_img.flatten())
    y_augments.append(y_train[i])
x_augments = np.array(x_augments, dtype='float32')
y_augments = np.array(y_augments)
x_train = np.concatenate((x_train, x_augments))
y_train = np.concatenate((y_train, y_augments))
print(f'Training set size:   {x_train.shape[0]}')
print(f'Validation set size: {x_val.shape[0]}')
# Training set dataloader
train_dataset = DatasetMNIST(x_train, y_train)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
# Validation set dataloader
val_dataset = DatasetMNIST(x_val, y_val)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)



# Evaluates a given network for the data in the given data loader and returns
# the accuracy (amount correct) as a percentage.
def accuracy(net, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            if(CUDA):
                images, labels = data[0].to(device), data[1].to(device)
            else:
                images, labels = data

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total


# For a given model and optimizer, trains using the data of the given data
# loader for the specified number of epochs.
def train(net, optimizer, loader, num_epochs, update_interval=100):
    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        # Adjust learning rate over time
        lr = 0.01 * 0.95**(NUM_EPOCHS + epoch)
        for g in optimizer.param_groups:
            g['lr'] = lr
        print(f'Learning rate: {lr:.6f}')

        running_loss = 0.0
        for i, data in enumerate(loader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            if(CUDA):
                inputs, labels = data[0].to(device), data[1].to(device)
            else:
                inputs, labels = data

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()

            # Print updates after a certain number of mini-batches
            if(i % update_interval == update_interval - 1):
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / update_interval))
                running_loss = 0.0
            
            if(i % BACKUP_INTERVAL == BACKUP_INTERVAL - 1):
                try:
                    torch.save(net, SAVE_FILE)
                except KeyboardInterrupt:
                    print('Finishing saving model...')
                    torch.save(net, SAVE_FILE)
                    print('Model saved.')

        # Print accuracy after each epoch
        train_accuracy = accuracy(net, trainloader)
        val_accuracy = accuracy(net, valloader)
        print('Train/Validation Accuracy: %.3f / %.3f' % (train_accuracy, val_accuracy))

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

# Model A - Logistic regression (Fully connected output, 0 hidden layers)
class NetA(nn.Module):
    def __init__(self):
        super(NetA, self).__init__()
        self.fc = nn.Linear(1 * 28 * 28, 10)

    def forward(self, x):
        x = x.view(x.size()[0], -1) # Reshape to be a single vector (per input)
        x = self.fc(x)
        return x

# Model B - 1 fully connected hidden layer
class NetB(nn.Module):
    def __init__(self, M):
        super(NetB, self).__init__()
        self.fc1 = nn.Linear(1 * 28 * 28, M)
        self.fc2 = nn.Linear(M, 10)

    def forward(self, x):
        x = x.view(x.size()[0], -1) # Reshape to be a single vector (per input)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Model C - Fully connected output, 1 convolutional layer with max pool
class NetC(nn.Module):
    def __init__(self, M, p, N):
        super(NetC, self).__init__()
        self.conv1 = nn.Conv2d(3, M, p)
        pool_size = int((29 - p) / N)
        self.pool = nn.MaxPool2d(pool_size, pool_size) # Equal window size and stride
        self.fc_input_size = M * N * N
        self.fc = nn.Linear(self.fc_input_size, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, self.fc_input_size)
        x = self.fc(x)
        return x




for i in range(HYPERPARAM_TESTS):
    # Choose hyperparameters
    #X = int(np.random.uniform(90, 200))
    #Y = int(np.random.uniform(50, 120))
    #M = int(np.random.uniform(50, 150))
    #N = 14
    #p = 5
    #lr = np.random.uniform(0.00001, 0.01)
    #m = np.random.uniform(0.4, 0.95)
    #lr = 0.002438
    #m = 0.484753
    #M = 324
    #print('Using lr=%f, m=%f, X=%d, Y=%d, M=%d, N=%d, p=%d' % (lr, m, X, Y, M, N, p))
    lr = 0
    m = 0.75


    # Make model
    net = Net(0, 0)
    #net = NetA() 
    #net = NetB(M)
    #net = NetC(M, p, N)

    if(CUDA):
        net.to(device)

    # If a previous model exists, load it
    if(False and os.path.isfile(SAVE_FILE)):
        print('Previous model "%s" found, loading...' % SAVE_FILE, end='')
        net = torch.load(SAVE_FILE)
        print('loaded.')

    # Define loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=m)

    # Train
    print('Starting training...')
    start = time()
    train(net, optimizer, trainloader, NUM_EPOCHS, UPDATE_INTERVAL)
    print(f'Finished training ({time() - start:.2f}s).')
    
    # Record hyperparameter results
    val_accuracy = accuracy(net, valloader)
    print(val_accuracy, lr, m)





# Display the accuracy for each class
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in valloader:
        if(CUDA):
            images, labels = data[0].to(device), data[1].to(device)
        else:
            images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))











