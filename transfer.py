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


# use the pretrained vgg19 CNN
vgg = models.vgg19(pretrained=True).features

# freeze vgg parameters, so were only updating the image not the weights
for param in vgg.parameters():
    param.requires_grad_(False)

# move the model to gpu if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vgg.to(device)

print(vgg)


