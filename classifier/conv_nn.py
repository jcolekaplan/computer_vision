"""
    Jacob Kaplan
    conv_nn.py
    
    Implement a neural network with convolutional layers along with fully
    connected layers
"""
import os
import sys
import pickle
import cv2 as cv
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import matplotlib.pyplot as plt

def resize(img):
    scale_percent = 50 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    return resized

def read_train_dir(train_dir):
    """
    Take in a directory of sub-directories of images
    Go through each sub-directory that matches one of the backgrounds
        (grass, ocean, readcarpet, road, wheatfield)
    Get all the images in each sub-directory and put it into a list
    Append that list of all images in sub-directory to train_dir_imgs list
    Return train_dir_imgs list which will contain all the images in all the
        sub-directories sorted by background classification
    """
    train_dir_imgs = list()
    train_dir_labels = list()
    img_dirs = ["grass", "ocean", "redcarpet", "road", "wheatfield"]
    for fil in os.listdir(train_dir):
        if fil in img_dirs:
            images, labels = find_images(train_dir + "/" + fil, fil)
            train_dir_imgs = train_dir_imgs + images
            train_dir_labels = train_dir_labels + labels
    return (train_dir_imgs, train_dir_labels)

def find_images(dir, bground):
    """
    Take in directory of images, open directory, read in each image,
    add to list of images, return list of images
    """
    images = list()
    labels = list()
    for fil in os.listdir(dir):
        if fil.lower().endswith('.jpeg'):
            try:
                img = cv.imread(dir + "/" + fil, 1)
                img = resize(img)
            except cv.error:
                print("{} malformed!".format(fil))
                sys.exit()
            images.append(img.flatten())
            labels.append(bground)
    return images, labels

def train_net(training_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(training_loader, 0):
            # get the inputs
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            
    print('Finished Training')
"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
"""



# Define Neural Network model
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(3 * 16 * 16, 64)
        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        #x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 3 * 16 *16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return(x)


if __name__ == "__main__":
    """
    Handle command line arguments
    """
    if len(sys.argv) != 3:
        print("Usage: {} load load_dir".format(sys.argv[0]))
        print("or Usage: {} train train_dir".format(sys.argv[0]))
        print("or Usage: {} test test_dir".format(sys.argv[0]))
        sys.exit()
    else:
        command = sys.argv[1].lower()
        dir_name = sys.argv[2].lower()

    commands = ["load", "train", "test"]
    if command not in commands:
        print("Command must be load, train, or test!")
        sys.exit()

    load_imgs = []
    if command == "load":
        """
        Used to convert a set of images in all sub-directories of a directory
            to vector and pickle them for later use. This can be used to load
            a training or testing set
        """
        try:
            load_imgs = read_train_dir(dir_name)
            outfile = open("{}/{}_set.pkl".format(dir_name, dir_name), "wb")
            pickle.dump(load_imgs, outfile)
            outfile.close()
        except ValueError:
            print("{} not found!".format(dir_name))
            sys.exit()

    elif command == "train":

        #infile = open("{}/{}_set.pkl".format(dir_name, dir_name), "rb")
        #training_set = pickle.load(infile)
        #infile.close()
        transform = transforms.Compose(
            [transforms.Resize((32,32)),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.ImageFolder(root='./train', transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

        testset = torchvision.datasets.ImageFolder(root='./test', transform=transform)
        testloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

        net = Net()
        train_net(trainloader)
        classes = ("road", "ocean", "redcarpet", "wheatfield", "grass")

        class_correct = list(0. for i in range(5))
        class_total = list(0. for i in range(5))
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]-1
                    class_correct[label] += c[i].item()
                    class_total[label] += 1


        for i in range(5):
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))
