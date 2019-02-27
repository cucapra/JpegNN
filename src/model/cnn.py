"""
Finetuning Torchvision Models
=============================

**Author:** `Nathan Inkawhich <https://github.com/inkawhich>`__

"""


######################################################################
# In this tutorial we will take a deeper look at how to finetune and
# feature extract the `torchvision
# models <https://pytorch.org/docs/stable/torchvision/models.html>`__, all
# of which have been pretrained on the 1000-class Imagenet dataset. This
# tutorial will give an indepth look at how to work with several modern
# CNN architectures, and will build an intuition for finetuning any
# PyTorch model. Since each model architecture is different, there is no
# boilerplate finetuning code that will work in all scenarios. Rather, the
# researcher must look at the existing architecture and make custom
# adjustments for each model.
# 
# In this document we will perform two types of transfer learning:
# finetuning and feature extraction. In **finetuning**, we start with a
# pretrained model and update *all* of the model’s parameters for our new
# task, in essence retraining the whole model. In **feature extraction**,
# we start with a pretrained model and only update the final layer weights
# from which we derive predictions. It is called feature extraction
# because we use the pretrained CNN as a fixed feature-extractor, and only
# change the output layer. For more technical information about transfer
# learning see `here <https://cs231n.github.io/transfer-learning/>`__ and
# `here <https://ruder.io/transfer-learning/>`__.
# 
# In general both transfer learning methods follow the same few steps:
# 
# -  Initialize the pretrained model
# -  Reshape the final layer(s) to have the same number of outputs as the
#    number of classes in the new dataset
# -  Define for the optimization algorithm which parameters we want to
#    update during training
# -  Run the training step
# 

from __future__ import print_function 
from __future__ import division
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
from jpeg_layer import *
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


######################################################################
# Inputs
# ------
# 
# Here are all of the parameters to change for the run. We will use the
# *hymenoptera_data* dataset which can be downloaded
# `here <https://download.pytorch.org/tutorial/hymenoptera_data.zip>`__.
# This dataset contains two classes, **bees** and **ants**, and is
# structured such that we can use the
# `ImageFolder <https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.ImageFolder>`__
# dataset, rather than writing our own custom dataset. Download the data
# and set the ``data_dir`` input to the root directory of the dataset. The
# ``model_name`` input is the name of the model you wish to use and must
# be selected from this list:
# 
# ::
# 
#    [resnet, alexnet, vgg, squeezenet, densenet, inception]
# 
# The other inputs are as follows: ``num_classes`` is the number of
# classes in the dataset, ``batch_size`` is the batch size used for
# training and may be adjusted according to the capability of your
# machine, ``num_epochs`` is the number of training epochs we want to run,
# and ``feature_extract`` is a boolean that defines if we are finetuning
# or feature extracting. If ``feature_extract = False``, the model is
# finetuned and all model parameters are updated. If
# ``feature_extract = True``, only the last layer parameters are updated,
# the others remain fixed.
# 

parser = argparse.ArgumentParser(description = \
        'Neural Network with JpegLayer')

# Top level data directory. Here we assume the format of the directory conforms 
#   to the ImageFolder structure
#data_dir = "./hymenoptera_data"
parser.add_argument('--data_dir', '-d', type=str,\
    default='/data/jenna/data/', \
    help='Directory of the input data. \
    String. Default: /data/jenna/data/')
# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
#model_name = "squeezenet"
parser.add_argument('--model_name', '-m', type=str,\
    default='squeezenet',\
    help = 'NN models to choose from [resnet, alexnet, \
    vgg, squeezenet, densenet, inception]. \
    String. Default: squeezenet')

# Number of classes in the dataset
#num_classes = 3
parser.add_argument('--num_classes', '-c', type=int,\
    default = 3,\
    help = 'Number of classes in the dataset. \
    Integer. Default: 3')
# Batch size for training (change depending on how much memory you have)
#batch_size = 8
parser.add_argument('--batch_size', '-b', type=int,\
    default = 8,\
    help = 'Batch size for training (can change depending\
    on how much memory you have. \
    Integer. Default: 8)')


# Number of epochs to train for 
#num_epochs = 25
parser.add_argument('-ep', '--num_epochs', type=int,\
    default = 25,\
    help = 'Number of echos to train for. \
    Integer. Default:25')

#Flag for whether to add jpeg layer to train quantization matrix
#add_jpeg_layer = True
parser.add_argument('--add_jpeg_layer', '-jpeg', \
    action = 'store_false',\
    help = 'Flag for adding jpeg layer to neural network. \
    Bool. Default: True')

#Flag for initialization for quantization table. When true,qtable is uniformly random. When false, qtable is jpeg standard.
parser.add_argument('--rand_qtable', '-rq', \
    action = 'store_false',\
    help='Flag for initialization for quantization table. \
    When true,qtable is uniformly random. When false, \
    qtable is jpeg standard.\
    Bool. Default: True.')

# Flag for feature extracting. When False, we finetune the whole model, 
#   when True we only update the reshaped layer params
#feature_extract = False
parser.add_argument('--feature_extract', '-f', \
    action = 'store_true',\
    help = 'Flag for feature extracting. When False, \
    we finetune the whole model.\
    Bool. Default: False.')

# Flag for printing trained quantization matrix
parser.add_argument('--qtable', '-q', \
    action = 'store_true',\
    help = 'Flag for print quantization matrix. \
    Bool. Default: False.')   

#Flag for visualizing the jpeg layer
parser.add_argument('--visualize', '-v',\
    action = 'store_false',\
    help = 'Flag for visualizing the jpeg layer. \
    Bool. Default: True')
#Flag for regularize the magnitude of quantization table
#regularize = True
parser.add_argument('--regularize','-r',\
    action = 'store_false',\
    help = 'Flag for regularize the magnitude of \
    quantizaiton table. Without the term, the quantization \
    table goes to 0 \
    Bool. Default: True')

train_quant_only = True

#parse the inputs
args,unparsed = parser.parse_known_args()
print(args)
######################################################################
# Helper Functions
# ----------------
# 
# Before we write the code for adjusting the models, lets define a few
# helper functions.
# 
# Model Training and Validation Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# The ``train_model`` function handles the training and validation of a
# given model. As input, it takes a PyTorch model, a dictionary of
# dataloaders, a loss function, an optimizer, a specified number of epochs
# to train and validate for, and a boolean flag for when the model is an
# Inception model. The *is_inception* flag is used to accomodate the
# *Inception v3* model, as that architecture uses an auxiliary output and
# the overall model loss respects both the auxiliary output and the final
# output, as described
# `here <https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958>`__.
# The function trains for the specified number of epochs and after each
# epoch runs a full validation step. It also keeps track of the best
# performing model (in terms of validation accuracy), and at the end of
# training returns the best performing model. After each epoch, the
# training and validation accuracies are printed.
# 

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    
                    #add regularization
                    
                    reg_loss = 0
                    factor = 0.01
                    if args.regularize:
                        reg_crit = nn.L1Loss(size_average=True)
                        #reg_crit = nn.BCELoss()
                        
                        target = torch.Tensor(3,8,8).cuda()
                        target.fill_(0)
                        for name, param in model.named_parameters():
                            if name == "0.quantize":
                                reg_loss = factor /reg_crit(param,target) * inputs.size(0)
                                break

                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    #print('orginal loss', loss)
                    #print('regularization', reg_loss)
                    loss =  reg_loss + loss
                    #print('after reg: ', loss)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        quantize = 0
                        for name, param in model.named_parameters():
                            if name == "0.quantize":
                                #print(param.grad[0][0][0])

                                #param.data = torch.round(param.data * 255)/255
                                #param.data.clamp_(-100/255,100/255)
                                #quantize = param.data
                                
                                #print(param.data[0][0]*255)
                         #   if "conv1.weight" in name:
                         #       print("1.features.0.w gradient!!!!!")
                         #       print(param.grad)
                         #       print("parameters!!!!!")
                         #       print(param)
                           # if name == "0.quantize_nograd":
                                #param.data = copy.deepcopy(quantize)
                                #print(param)
                                break

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


######################################################################
# Set Model Parameters’ .requires_grad attribute
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# This helper function sets the ``.requires_grad`` attribute of the
# parameters in the model to False when we are feature extracting. By
# default, when we load a pretrained model all of the parameters have
# ``.requires_grad=True``, which is fine if we are training from scratch
# or finetuning. However, if we are feature extracting and only want to
# compute gradients for the newly initialized layer then we want all of
# the other parameters to not require gradients. This will make more sense
# later.
# 

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for name, param in model.named_parameters():
            param.requires_grad = False


######################################################################
# Initialize and Reshape the Networks
# -----------------------------------
# 
# Now to the most interesting part. Here is where we handle the reshaping
# of each network. Note, this is not an automatic procedure and is unique
# to each model. Recall, the final layer of a CNN model, which is often
# times an FC layer, has the same number of nodes as the number of output
# classes in the dataset. Since all of the models have been pretrained on
# Imagenet, they all have output layers of size 1000, one node for each
# class. The goal here is to reshape the last layer to have the same
# number of inputs as before, AND to have the same number of outputs as
# the number of classes in the dataset. In the following sections we will
# discuss how to alter the architecture of each model individually. But
# first, there is one important detail regarding the difference between
# finetuning and feature-extraction.
# 
# When feature extracting, we only want to update the parameters of the
# last layer, or in other words, we only want to update the parameters for
# the layer(s) we are reshaping. Therefore, we do not need to compute the
# gradients of the parameters that we are not changing, so for efficiency
# we set the .requires_grad attribute to False. This is important because
# by default, this attribute is set to True. Then, when we initialize the
# new layer and by default the new parameters have ``.requires_grad=True``
# so only the new layer’s parameters will be updated. When we are
# finetuning we can leave all of the .required_grad’s set to the default
# of True.
# 
# Finally, notice that inception_v3 requires the input size to be
# (299,299), whereas all of the other models expect (224,224).
# 
# Resnet
# ~~~~~~
# 
# Resnet was introduced in the paper `Deep Residual Learning for Image
# Recognition <https://arxiv.org/abs/1512.03385>`__. There are several
# variants of different sizes, including Resnet18, Resnet34, Resnet50,
# Resnet101, and Resnet152, all of which are available from torchvision
# models. Here we use Resnet18, as our dataset is small and only has two
# classes. When we print the model, we see that the last layer is a fully
# connected layer as shown below:
# 
# ::
# 
#    (fc): Linear(in_features=512, out_features=1000, bias=True) 
# 
# Thus, we must reinitialize ``model.fc`` to be a Linear layer with 512
# input features and 2 output features with:
# 
# ::
# 
#    model.fc = nn.Linear(512, num_classes)
# 
# Alexnet
# ~~~~~~~
# 
# Alexnet was introduced in the paper `ImageNet Classification with Deep
# Convolutional Neural
# Networks <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`__
# and was the first very successful CNN on the ImageNet dataset. When we
# print the model architecture, we see the model output comes from the 6th
# layer of the classifier
# 
# ::
# 
#    (classifier): Sequential(
#        ...
#        (6): Linear(in_features=4096, out_features=1000, bias=True)
#     ) 
# 
# To use the model with our dataset we reinitialize this layer as
# 
# ::
# 
#    model.classifier[6] = nn.Linear(4096,num_classes)
# 
# VGG
# ~~~
# 
# VGG was introduced in the paper `Very Deep Convolutional Networks for
# Large-Scale Image Recognition <https://arxiv.org/pdf/1409.1556.pdf>`__.
# Torchvision offers eight versions of VGG with various lengths and some
# that have batch normalizations layers. Here we use VGG-11 with batch
# normalization. The output layer is similar to Alexnet, i.e.
# 
# ::
# 
#    (classifier): Sequential(
#        ...
#        (6): Linear(in_features=4096, out_features=1000, bias=True)
#     )
# 
# Therefore, we use the same technique to modify the output layer
# 
# ::
# 
#    model.classifier[6] = nn.Linear(4096,num_classes)
# 
# Squeezenet
# ~~~~~~~~~~
# 
# The Squeeznet architecture is described in the paper `SqueezeNet:
# AlexNet-level accuracy with 50x fewer parameters and <0.5MB model
# size <https://arxiv.org/abs/1602.07360>`__ and uses a different output
# structure than any of the other models shown here. Torchvision has two
# versions of Squeezenet, we use version 1.0. The output comes from a 1x1
# convolutional layer which is the 1st layer of the classifier:
# 
# ::
# 
#    (classifier): Sequential(
#        (0): Dropout(p=0.5)
#        (1): Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1))
#        (2): ReLU(inplace)
#        (3): AvgPool2d(kernel_size=13, stride=1, padding=0)
#     ) 
# 
# To modify the network, we reinitialize the Conv2d layer to have an
# output feature map of depth 2 as
# 
# ::
# 
#    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
# 
# Densenet
# ~~~~~~~~
# 
# Densenet was introduced in the paper `Densely Connected Convolutional
# Networks <https://arxiv.org/abs/1608.06993>`__. Torchvision has four
# variants of Densenet but here we only use Densenet-121. The output layer
# is a linear layer with 1024 input features:
# 
# ::
# 
#    (classifier): Linear(in_features=1024, out_features=1000, bias=True) 
# 
# To reshape the network, we reinitialize the classifier’s linear layer as
# 
# ::
# 
#    model.classifier = nn.Linear(1024, num_classes)
# 
# Inception v3
# ~~~~~~~~~~~~
# 
# Finally, Inception v3 was first described in `Rethinking the Inception
# Architecture for Computer
# Vision <https://arxiv.org/pdf/1512.00567v1.pdf>`__. This network is
# unique because it has two output layers when training. The second output
# is known as an auxiliary output and is contained in the AuxLogits part
# of the network. The primary output is a linear layer at the end of the
# network. Note, when testing we only consider the primary output. The
# auxiliary output and primary output of the loaded model are printed as:
# 
# ::
# 
#    (AuxLogits): InceptionAux(
#        ...
#        (fc): Linear(in_features=768, out_features=1000, bias=True)
#     )
#     ...
#    (fc): Linear(in_features=2048, out_features=1000, bias=True)
# 
# To finetune this model we must reshape both layers. This is accomplished
# with the following
# 
# ::
# 
#    model.AuxLogits.fc = nn.Linear(768, num_classes)
#    model.fc = nn.Linear(2048, num_classes)
# 
# Notice, many of the models have similar output structures, but each must
# be handled slightly differently. Also, check out the printed model
# architecture of the reshaped network and make sure the number of output
# features is the same as the number of classes in the dataset.
# 


def initialize_model(model_name, num_classes, feature_extract, add_jpeg_layer = False, train_quant_only = False, rand_qtable = True, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224
    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()
    
    if add_jpeg_layer:
        if train_quant_only:
            model_ft.load_state_dict(torch.load("model.final"))
        model_ft = nn.Sequential(JpegLayer( \
                   rand_qtable = rand_qtable),\
                   model_ft)
    return model_ft, input_size



# Initialize the model for this run
model_ft, input_size = initialize_model(args.model_name, args.num_classes, args.feature_extract, args.add_jpeg_layer, train_quant_only, args.rand_qtable, use_pretrained=True)
    
# Print the model we just instantiated
print(model_ft) 


######################################################################
# Load Data
# ---------
# 
# Now that we know what the input size must be, we can initialize the data
# transforms, image datasets, and the dataloaders. Notice, the models were
# pretrained with the hard-coded normalization values, as described
# `here <https://pytorch.org/docs/master/torchvision/models.html>`__.
# 

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_dir, x), data_transforms[x]) for x in ['train', 'val']}

dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


######################################################################
# Create the Optimizer
# --------------------
# 
# Now that the model structure is correct, the final step for finetuning
# and feature extracting is to create an optimizer that only updates the
# desired parameters. Recall that after loading the pretrained model, but
# before reshaping, if ``feature_extract=True`` we manually set all of the
# parameter’s ``.requires_grad`` attributes to False. Then the
# reinitialized layer’s parameters have ``.requires_grad=True`` by
# default. So now we know that *all parameters that have
# .requires_grad=True should be optimized.* Next, we make a list of such
# parameters and input this list to the SGD algorithm constructor.
# 
# To verify this, check out the printed parameters to learn. When
# finetuning, this list should be long and include all of the model
# parameters. However, when feature extracting this list should be short
# and only include the weights and biases of the reshaped layers.
# 

# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are 
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
name_to_update = []
params_to_update = [] #model_ft.parameters()
params_quantize = []
name_quantize = []
print("Params to learn:")
#if args.feature_extract:
#    params_to_update = []
for name,param in model_ft.named_parameters():
    if param.requires_grad == True:
        if train_quant_only and \
         name=="0.quantize":
            params_quantize.append(param)
            name_quantize.append(name)
        else:
            params_to_update.append(param)
            name_to_update.append(name)
            
#else:
#    for name,param in model_ft.named_parameters():
#        if param.requires_grad == True:
#            print("\t",name)

# Observe that all parameters are being optimized
if train_quant_only:
    optimizer_ft = optim.Adam(params_quantize, lr = 0.0005)
    print(name_quantize)
else:
    optimizer_ft = optim.SGD(params_to_update, lr = 0.001, momentum=0.9)
    print(name_to_update)
       # optim.SGD([{'params': params_to_update},\
       # {'params': params_quantize, 'lr': 0.005, 'momentum':0.9}], lr=0.0005, momentum=0.9)



######################################################################
# Run Training and Validation Step
# --------------------------------
# 
# Finally, the last step is to setup the loss for the model, then run the
# training and validation function for the set number of epochs. Notice,
# depending on the number of epochs this step may take a while on a CPU.
# Also, the default learning rate is not optimal for all of the models, so
# to achieve maximum accuracy it would be necessary to tune for each model
# separately.
# 

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=args.num_epochs, is_inception=(args.model_name=="inception"))
if args.add_jpeg_layer == False:
    torch.save(model_ft.state_dict(), "model.final")

#print the trained quantization matrix
if args.qtable:
    print('--------- the trained quantize table ---------')
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True and\
                name == "0.quantize":
            print('Y',param.data[0]*255)
            print('Cb',param.data[1]*255)
            print('Cr',param.data[2]*255)




# Let's visualize feature maps after jpeg layer
def get_activation(name):
   def hook(model, input, output):
       activation[name] = output.detach()
   return hook

if args.add_jpeg_layer:
    activation = {}
    
    model_ft[0].register_forward_hook(get_activation('0.JpegLayer'))



    data, _ = image_datasets["train"][0]
    
    fig, axarr = plt.subplots(2)
    f1 = data.cpu().data.numpy()
    f1 = (np.transpose(f1,(1,2,0))*255).astype(np.uint8)
    axarr[0].imshow(f1)
    
    data.unsqueeze_(0)
    output = model_ft(data.to(device))
    
    f2 = activation['0.JpegLayer'].squeeze().cpu().data.numpy()
    f2 = (np.transpose(f2, (1,2,0))*255).astype(np.uint8)
    axarr[1].imshow(f2)
    if args.visualize: 
        plt.show()
    
    #save images
    from psnr import psnr, compressJ, save
    from PIL import Image
    save(f1, "org.bmp")
    save(f2, "myJpeg.jpg")

###############################
##### standard python jpeg ####
###############################
#im = compressJ(f1,"toJpeg.jpg")
#im = np.array(im, np.int16).transpose(2,0,1)
#
##############################
#####         psnr        #### 
##############################
#f1 = np.array(f1,np.int16).transpose(2,0,1)
#f2 = np.array(f2,np.int16).transpose(2,0,1)
#print("compression results!")
#print("PSNR - my jpeg: ", psnr(f2[0],f1[0]))
#print("PSNR - PIL jpeg", psnr(im[0], f1[0]))
#print("PSNR - my vs. PIL", psnr(im[0], f2[0]))



#######################################################################
## Comparison with Model Trained from Scratch
## ------------------------------------------
## 
## Just for fun, lets see how the model learns if we do not use transfer
## learning. The performance of finetuning vs. feature extracting depends
## largely on the dataset but in general both transfer learning methods
## produce favorable results in terms of training time and overall accuracy
## versus a model trained from scratch.
## 
#
#
## Initialize the non-pretrained version of the model used for this run
#scratch_model,_ = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False)
#scratch_model = scratch_model.to(device)
#scratch_optimizer = optim.SGD(scratch_model.parameters(), lr=0.001, momentum=0.9)
#scratch_criterion = nn.CrossEntropyLoss()
#_,scratch_hist = train_model(scratch_model, dataloaders_dict, scratch_criterion, scratch_optimizer, num_epochs=num_epochs, is_inception=(model_name=="inception"))
#
## Plot the training curves of validation accuracy vs. number 
##  of training epochs for the transfer learning method and
##  the model trained from scratch
#ohist = []
#shist = []
#
#ohist = [h.cpu().numpy() for h in hist]
#shist = [h.cpu().numpy() for h in scratch_hist]
#
#plt.title("Validation Accuracy vs. Number of Training Epochs")
#plt.xlabel("Training Epochs")
#plt.ylabel("Validation Accuracy")
#plt.plot(range(1,num_epochs+1),ohist,label="Pretrained")
#plt.plot(range(1,num_epochs+1),shist,label="Scratch")
#plt.ylim((0,1.))
#plt.xticks(np.arange(1, num_epochs+1, 1.0))
#plt.legend()
#plt.show()
#

######################################################################
# Final Thoughts and Where to Go Next
# -----------------------------------
# 
# Try running some of the other models and see how good the accuracy gets.
# Also, notice that feature extracting takes less time because in the
# backward pass we do not have to calculate most of the gradients. There
# are many places to go from here. You could:
# 
# -  Run this code with a harder dataset and see some more benefits of
#    transfer learning
# -  Using the methods described here, use transfer learning to update a
#    different model, perhaps in a new domain (i.e. NLP, audio, etc.)
# -  Once you are happy with a model, you can export it as an ONNX model,
#    or trace it using the hybrid frontend for more speed and optimization
#    opportunities.
# 

