import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import time

training_data = np.load("catsdogsdata80.npy", allow_pickle = True)
# Always make sure it matches the IMG_SIZE with which the data was built
IMG_SIZE = 80

tb = SummaryWriter()

class Net(nn.Module):
    def __init__(self):
        
        super(Net, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        # Flatten layer is neccessary between a convolutional and a linear layer
        self.flatten = nn.Flatten()
        # We have a huge number of neurons because of high resolution, so it's a good idea to dropout
        self.dropout = nn.Dropout(0.85)
        # Linear layers
        #self.fc1 = nn.Linear(18496, 2048)
        self.fc1 = nn.Linear(9248, 2048)
        self.fc2 = nn.Linear(2048, 128)
        self.fc3 = nn.Linear(128, 2)
        
    def forward(self, x):
        # Passing data through convoltional layers and ReLu activation function
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # Flattening data
        x = self.flatten(x)
        x = F.relu(x)
        # Dropout
        x = self.dropout(x)
        # Linear layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        # Activation function
        result = F.log_softmax(x, dim = 1)
        
        return result

def fwd_pass(X, y, train = False):
    # Zero the gradient when training
    # We need to do this because PyTorch accumulates the gradients on subsequent backward passes
    if train:
        net.zero_grad()
    # Pass the data through the network
    outputs = net(X)
    # Count accuracy and loss
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
    acc = matches.count(True)/len(matches)
    loss = loss_function(outputs, y)
    # Backpropagation when training
    if train:
        loss.backward()
        optimizer.step()
    
    return acc, loss

def test(size = 32):
    # Random start makes sure we don't test on the same data
    random_start = np.random.randint(len(test_X - size))
    X, y = test_X[random_start:random_start + size], test_y[random_start: random_start + size]
    # We don't update the gradients when testing
    with torch.no_grad():
        val_acc, val_loss = fwd_pass(X.view(-1, 1, IMG_SIZE, IMG_SIZE).to(device), y.to(device))
        
    return val_acc, val_loss
    
# Initializing the nn object    
net = Net()
# Defining optimizer and loss function
optimizer = optim.Adam(net.parameters(), lr = 0.00003) 
loss_function = nn.MSELoss()

# Converting a NumPy array of images into a PyTorch Tensor
X = torch.Tensor([i[0] for i in training_data]).view(-1, IMG_SIZE, IMG_SIZE)
# Converting the pixel values from [0:255] to [0:1]
# Neural networks in general work better with the latter
X = X/255.0
# Converting a NumPy array of labels into a PyTorch Tensor
y = torch.Tensor([i[1] for i in training_data])
# Specifies which percentage of data is used for testing
VAL_PCT = 0.1
val_size = int(len(X)*VAL_PCT)

# Splits the data into a training and testing sets
train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]
# Frees up memory
training_data = None
X = None
y = None
# Makes sure the nn is in training mode
net.train()
# Computing on a Nvidia CUDA card if the machine has one
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    net.to(device)


def train():
    BATCH_SIZE = 600
    EPOCHS = 100
    # Model name can be used to save models each epoch and picking the best
    MODEL_NAME = f"model-{EPOCHS}-epochs-{BATCH_SIZE}-batch-{int(time.time())}"
    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
            # Taking a batch of data
            batch_X = train_X[i:i + BATCH_SIZE].view(-1, 1, IMG_SIZE, IMG_SIZE)
            batch_y = train_y[i:i + BATCH_SIZE]
            # Transfering it to a GPU if possible
            if torch.cuda.is_available():
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            # Passing the data through the network, storing accuracy and loss
            acc, loss = fwd_pass(batch_X, batch_y, train = True)
        
        # Testing the model on the test data
        net.eval()
        val_acc, val_loss = test(1000)
        net.train()
        # Monitoring the results with TensorBoard    
        tb.add_scalar('train_loss', loss, epoch)
        tb.add_scalar('train_acc', acc, epoch)
        tb.add_scalar('val_loss', val_loss, epoch)
        tb.add_scalar('val_acc', val_acc, epoch)
    
# Training the network
train()
print("TRAINING FINISHED")
# Testing the network on all testing data
val_acc, val_loss = test(size = len(test_X))
print(val_acc, val_loss)
