import argparse

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
import torch.nn.functional as F

from PIL import Image

from collections import OrderedDict

import time

import numpy as np
import matplotlib.pyplot as plt
#utilites is the .py file I have created containing save_checkpoint and load_checkpoint functions
from utilities import save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Parser to train a Neural Network using transfer learning")
    parser.add_argument('--data_directory', dest = 'data_dir', action='store')
    parser.add_argument('--arch', dest='arch', default='densenet121', choices=['vgg13', 'densenet121'])
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001')
    parser.add_argument('--hidden_units', dest='hidden_units', default='512')
    parser.add_argument('--epochs', dest='epochs', default='3')
    parser.add_argument('--gpu', dest = 'gpu', action='store', default='gpu')
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth")
    return parser.parse_args()

#To convert the user provided input for gpu as an input to the train function,
#I need to define a variable first, then call the attribute of gpu I have created
input_args = parse_args()

criteterion = nn.NLLLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=0.01)
model = input_args.arch
gpu = input_args.gpu
epochs = input_args.epochs
directory = input_args.data_dir

def train(model, criterion, optimizer, train_dataloader, valid_dataloader, epochs, gpu):
    #Define global variables we will be using during the training
    #epochs are times algorithm backpropagates
    #epochs = 5 #in the notebook I have set it 5 - here I get it from arguments set in main
    steps = 0 #we will track the number of trainsteps we do so we set this to zero
    print_every = 50
    if gpu = 'gpu':
        device = 'cuda'
    else:
        device = 'cpu'
    for epoch in range(epochs):
        #Define local variables we will be using during the training
        training_loss = 0 #we wil track the training loss we do so we set this to zero
        for images, labels in train_dataloader: #loop into our data (images)
            steps += 1 #cumulate steps aka batches. Every time we go through one of these batches, we will increment batch steps here
            images, labels = images.to(device), labels.to(device)# Move input and label tensors to the default device
            optimizer.zero_grad() #Here we zero our gradients in order not to have leftover values from previous loops
            log_ps = model.forward(images) #get our log probabilities from our model
            loss = criterion(log_ps, labels) #We get the loss from the criterion
            loss.backward() #We do a backwards pass
            optimizer.step() #With the optimizer we take a step
            training_loss += loss.item() #Here we increment our training loss - This is where we keep track of our training loss


### This is where our training loop ends ###
### Now we drop out of our training loop to test our network's accuracy and loss on our validation dataset ###

            if steps % print_every == 0: #if we have run out of steps - aka batches -
                #initialize local function variables
                valid_loss = 0
                accuracy = 0
                model.eval() # this turns our model to the evaluation inference mode which turns off Dropout
                # In this way we can accurately use our network for making predictions
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    for images, labels in valid_dataloader:
                        images, labels = images.to(device), labels.to(device)
                        output = model.forward(images)
                        #images, labels = images.to(device), labels.to(device) #transfer my tensors to the GPU
                        og_ps = model(images) #with the model we already have we will pass images from our validation dataset
                        valid_loss += criterion(log_ps, labels) #get loss from validation dataset
                        #calculate accuracy
                        ps = torch.exp(log_ps) #our model returns log probabilities, so to get crude probabilities we do this
                        top_p, top_class = ps.topk(1, dim=1) #here we get top probabilities and top classes using ps.topk()
                        equality = top_class == labels.view(*top_class.shape) #check for equality with our labels
                        accuracy += torch.mean(equality.type(torch.FloatTensor)) #with equality tensor we update our accuracy
                print(f"Epoch {epoch+1}/{epochs}.. "
                f"Train loss: {training_loss/print_every:.3f}.. "
                f"Valid loss: {valid_loss/len(valid_dataloader):.3f}.. "
                f"Valid accuracy: {accuracy/len(valid_dataloader):.3f}")

                training_loss = 0 #we set our training loss back to zero
                model.train() #we set our model in training mode - we now use the dropout



def main():
    print("Start training") # message informing about the begining of the training phase
    args = parse_args()

    # TODO: Define your transforms for the training, validation, and testing sets
    training_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                               transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    validaiton_data_transforms = transforms.Compose([transforms.Resize(256),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    test_data_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    # TODO: Load the datasets with ImageFolder
    train_image_dataset = datasets.ImageFolder(train_dir, transform = training_data_transforms)
    valid_image_dataset = datasets.ImageFolder(valid_dir, transform = validaiton_data_transforms)
    test_image_dataset = datasets.ImageFolder(test_dir, transform = test_data_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_image_dataset, batch_size = 32, shuffle = True)
    valid_dataloader = torch.utils.data.DataLoader(valid_image_dataset, batch_size = 32, shuffle = True)
    test_dataloader = torch.utils.data.DataLoader(test_image_dataset, batch_size = 32, shuffle = True)

    # Use GPU if it's available
    # Good coding practice - do this after the transformation process
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_names = train_image_dataset.classes

    model = getattr(models, args.arch)(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    if args.arch == "vgg13":
        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
                                    ('fc1', nn.Linear(25088, 1500)),
                                    ('relu', nn.ReLU()),
                                    ('dropout',nn.Dropout(.2)),
                                    ('fc2', nn.Linear(1500,102)),
                                    ('output', nn.LogSoftmax(dim=1))
                                    ]))
    elif args.arch == "densenet121":
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(1024, 500)),
                                  ('drop', nn.Dropout(p=0.6)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(500, 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    class_index = train_image_dataset.class_to_idx
    train(model, criterion, optimizer, dataloaders, epochs, gpu)
    model.class_to_idx = class_index
    path = args.save_dir
    save_checkpoint(path, model, optimizer, args, classifier)


if __name__ == "__main__":
    main()
