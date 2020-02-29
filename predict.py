import argparse

import torch
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F

import numpy as np

from PIL import Image

import json
import os
import random

from utils import load_checkpoint, load_cat_names

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', action='store', default='checkpoint.pth')
    parser.add_argument('--top_k', dest='top_k', default='3')
    parser.add_argument('--filepath', dest='filepath', default='flowers/test/1/image_06743.jpg') # use a deafault filepath to a primrose image
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store', default='gpu')
    return parser.parse_args()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_pil = Image.open(image)
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = adjustments(img_pil)

    return img_tensor

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        image = process_image(image_path)


        #The model requires a 4-D tensor:
        #1. first dimension specifying the images in a batch,
        #2. second dimension specifying the number of color channels
        #3. third dimension image width
        #4. fourth dimension immage height
        #we should add the batch dimension for the image tensor.
        #we use image = image.unsqueeze(0) This will add one more dimension at index 0.
        image = image.unsqueeze(0)
        #image = image.unsqueeze_(0) #unsqueeze with replacement

        image = image.float()
        image = image.to(device)
        logps = model.forward(image)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(5, dim=1)
        top_p = top_p.tolist()[0]
        top_class = top_class.tolist()[0]

        idx_to_class = {model.class_to_idx[i]: i for i in model.class_to_idx}
        labels = []
    for c in top_class:
        labels.append(cat_to_name[idx_to_class[c]])

    return top_p, labels

def main():
    args = parse_args()
    gpu = args.gpu
    model = load_checkpoint(args.checkpoint)
    cat_to_name = load_cat_names(args.category_names)

    img_path = args.filepath
    probs, classes = predict(img_path, model, int(args.top_k), gpu)
    labels = [cat_to_name[str(index)] for index in classes]
    probability = probs
    print('File selected: ' + img_path)

    print(labels)
    print(probability)

    i=0 # this prints out top k classes and probs as according to user
    while i < len(labels):
        print("{} with a probability of {}".format(labels[i], probability[i]))
        i += 1 # cycle through

if __name__ == "__main__":
    main()
