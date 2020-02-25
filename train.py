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
    parser.add_argument('--data_directory', dest = 'data_directory', action='store')
    parser.add_argument('--arch', dest='arch', default='densenet121', choices=['vgg13', 'densenet121'])
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001')
    parser.add_argument('--hidden_units', dest='hidden_units', default='512')
    parser.add_argument('--epochs', dest='epochs', default='3')
    parser.add_argument('--gpu', dest = 'gpu', action='store', default='gpu')
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth")
    return parser.parse_args()


input_args = parse_args()

model = input_args.arch
gpu = input_args.gpu
epochs = input_args.epochs
criterion =input_args.

def train(model, criterion, optimizer, train_dataloader, epochs, gpu):
    #Define global variables we will be using during the training
    #epochs are times algorithm backpropagates
    #epochs = 5 #in the notebook I have set it 5 - here I get it from arguments set in main
    steps = 0 #we will track the number of trainsteps we do so we set this to zero
    print_every = 50
    for epoch in range(epochs):
        #Define local variables we will be using during the training
        training_loss = 0 #we wil track the training loss we do so we set this to zero
        for images, labels in train_dataloader: #loop into our data (images)
            steps += 1 #cumulate steps aka batches. Every time we go through one of these batches, we will increment batch steps here
            if gpu = 'gpu':
                device = 'cuda'
            else:
                device = 'cpu'
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
