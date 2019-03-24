from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms, models

def load_image(path, max_size=400, shape=None):
    image = Image.open(path).convert('RGB')

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),
                                              (0.229, 0.224, 0.225))])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0)

    return image


# helper function for un-normalizing an image 
# and converting it from a Tensor image to a NumPy image for display
def im_convert(tensor):
    image = tensor.to('cpu').clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + \
                    np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image


def get_features(image, model, layers=None):

    if layers is None:
        layers = {'0': 'conv1_1',  # style
                  '5': 'conv2_1',  # style
                  '10': 'conv3_1', # style
                  '21': 'conv4_2'} # content
    
    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        if name in layers:
            x = layer(x)
            features[layers[name]] = x

    return features


def gram_matrix(tensor):
    _, d, h, w = tensor.shape
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())

    return gram


# use the pretrained vgg19 CNN
vgg = models.vgg19(pretrained=True).features

# freeze vgg parameters, so were only updating the image not the weights
for param in vgg.parameters():
    param.requires_grad_(False)

# move the model to gpu if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vgg.to(device)

# load in content and style image
content = load_image('images/janelle.png').to(device)
# resize style image to match content image
style = load_image('images/delaunay.jpg', shape=content.shape[-2:]).to(device)

# get desired content and style features
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# calculate gram matrices for style representation layers
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# prepare a copy of the content image that will be iteratively altered
target = content.clone().requires_grad_(True).to(device)

# set weights to fiddle with the way the style is transfered
style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.8,
                 'conv3_1': 0.5}

content_weight = 1
style_weight = 3e6

