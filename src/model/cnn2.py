"""
Finetuning Torchvision Models
=============================

**Author:** `Nathan Inkawhich <https://github.com/inkawhich>`__

"""



from __future__ import print_function 
from __future__ import division
import sys
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
#
def parse_args(args):
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    
    parser.add_argument('--data_dir', '-d', type=str,\
        default='/data/jenna/data/', \
        help='Directory of the input data. \
        String. Default: /data/jenna/data/')
    parser.add_argument('--model_name', '-m', type=str,\
        default='squeezenet',\
        help = 'NN models to choose from [resnet, alexnet, \
        vgg, squeezenet, densenet, inception]. \
        String. Default: squeezenet')
    
    parser.add_argument('--num_classes', '-c', type=int,\
        default = 3,\
        help = 'Number of classes in the dataset. \
        Integer. Default: 3')
    
    parser.add_argument('--batch_size', '-b', type=int,\
        default = 8,\
        help = 'Batch size for training (can change depending\
        on how much memory you have. \
        Integer. Default: 8)')
    
    
    parser.add_argument('-ep', '--num_epochs', type=int,\
        default = 25,\
        help = 'Number of echos to train for. \
        Integer. Default:25')
    
    parser.add_argument('--add_jpeg_layer', '-jpeg', \
        action = 'store_false',\
        help = 'Flag for adding jpeg layer to neural network. \
        Bool. Default: True')
    
    parser.add_argument('--rand_qtable', '-rq', \
        action = 'store_false',\
        help='Flag for initialization for quantization table. \
        When true,qtable is uniformly random. When false, \
        qtable is jpeg standard.\
        Bool. Default: True.')
    
    parser.add_argument('--qtable', '-q', \
        action = 'store_true',\
        help = 'Flag for print quantization matrix. \
        Bool. Default: False.')   
    
    parser.add_argument('--visualize', '-v',\
        action = 'store_false',\
        help = 'Flag for visualizing the jpeg layer. \
        Bool. Default: True')
    parser.add_argument('--regularize','-r',\
        action = 'store_false',\
        help = 'Flag for regularize the magnitude of \
        quantizaiton table. Without the term, the quantization \
        table goes to 0 \
        Bool. Default: True')
    parser.add_argument('--gpu_id', type=str,\
        default = '0',\
        help = 'Specify GPU id')
    
    parser.add_argument('--quality', type = int,\
        default = 50,\
        help = 'Jpeg quality. It is used to calculate \
        a quality factor for different compression rate.  \
        Integer. Default: 50')
    parser.add_argument('--quant_only', action = 'store_true')
    parser.add_argument('--cnn_only', action = 'store_true')
    
    args,unparsed = parser.parse_known_args()
    args.feature_extract = False
    # args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.iftrain = not(args.quant_only and args.cnn_only)
    args.dir = os.path.dirname(__file__)
    print(args)
    return args
#####################################################################~~~~~~~~~~~~~~~~~~~~
## training and validation.
# 

def train_model(args, model, dataloaders, criterion, optimizer, is_inception=False):
    since = time.time()

    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    phases =['train', 'val'] 
    if not args.iftrain:
        phases = ['val']
    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)
                # zero the parameter gradients
                if args.iftrain:
                    optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        if args.add_jpeg_layer:
                            outputs,means,_ = model(inputs)
                        else:
                            outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        if args.regularize and args.add_jpeg_layer:
                            reg_crit = nn.L1Loss(reduction='mean')
                            target = torch.zeros([8,8]).to(args.device)
                            factor = 0.02
                            reg_loss = factor*reg_crit(means,target)
                            #loss = loss + reg_loss
                    _, preds = torch.max(outputs, 1)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        # qt_grad = getattr(model,'0').LQ.qtable.grad
                        # print(qt_grad, qt_grad.shape)
                        # raise Exception('ddddd')
                        optimizer.step()
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
## 
def set_parameter_requires_grad(model, first, feature_extract, quant_only=False, cnn_only=False):
    if first and feature_extract:
        quant_only = True
        cnn_only = True
    for name, param in model.named_parameters():
        if  (quant_only and 'quantize' not in name) or\
            (cnn_only and 'quantize' in name):
            param.requires_grad = False


######################################################################
# Initialize and Reshape the Networks

def initialize_model(args, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if args.model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft,\
           True, args.feature_extract)       
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, args.num_classes)
        input_size = 224

    elif args.model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft,\
           True, args.feature_extract)       
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, args.num_classes)
        input_size = 224

    elif args.model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft,\
           True, args.feature_extract)    
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, args.num_classes)
        input_size = 224

    elif args.model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft,\
           True, args.feature_extract)       
        model_ft.classifier[1] = nn.Conv2d(512, args.num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = args.num_classes
        input_size = 224

    elif args.model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft,\
           True, fargs.eature_extract)       
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, args.num_classes) 
        input_size = 224

    elif args.model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft,\
           True, args.feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, args.num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,args.num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()
    
    if args.add_jpeg_layer:
        print('add jpeg layer!')
        load_from = os.path.join(args.dir,"model.final")
        if args.quant_only and not args.cnn_only:
            model_ft.load_state_dict(torch.load(load_from))
        if not args.iftrain:
            model_ft.load_state_dict(torch.load(load_from))
        model_ft = sequential(JpegLayer( \
                   rand_qtable = args.rand_qtable, cnn_only = args.cnn_only, quality = args.quality),\
                   model_ft)
    set_parameter_requires_grad(model_ft,\
           False, args.feature_extract,
           args.quant_only, args.cnn_only)
    print(model_ft) 
    model_ft = model_ft.to(args.device)
    return model_ft, input_size


######################################################################
# Load Data
# ---------

def load_data(args, input_size):
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
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
    return image_datasets, dataloaders_dict
# Detect if we have a GPU available

######################################################################
# Create the Optimizer
def create_optimizer(args, model_ft):

    params_to_update = [] #model_ft.parameters()
    params_quantize = []
    print("Params to learn:")
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            if 'quantize' in name:
                params_quantize.append(param)
                print('quantize\t', name)
            else:
                params_to_update.append(param)
                print('net\t',name)
                
    if args.iftrain:
        optimizer_ft = optim.SGD(
            [{'params': params_to_update}, {'params': params_quantize, 'lr': 0.00001} ], 
            lr = 0.0005, momentum=0.9)
    else:
        optimizer_ft = None
        # optim.SGD([{'params': params_to_update},\
           # {'params': params_quantize, 'lr': 0.005, 'momentum':0.9}], lr=0.0005, momentum=0.9)
    return optimizer_ft

def summarize(args, model_ft, image_datasets):
    #print the trained quantization matrix
    if args.qtable:
        print('--------- the trained quantize table ---------')
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True and\
                    "quantize" in name:
                print(name, param.data*255)

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output[0].detach()
        return hook   
    if args.add_jpeg_layer:
        activation = {}
        model_ft[0].register_forward_hook(get_activation('0.JpegLayer'))
    
        data, _ = image_datasets["val"][0]
        f1 = data.cpu().data.numpy()
        f1 = (np.transpose(f1,(1,2,0))*255).astype(np.uint8)
        data.unsqueeze_(0)
        output = model_ft(data.to(args.device))
        f2 = activation['0.JpegLayer'].squeeze().cpu().data.numpy()
        f2 = (np.transpose(f2, (1,2,0))*255).astype(np.uint8)
        if args.visualize: 
            fig, axarr = plt.subplots(2)
            axarr[0].imshow(f1)
            axarr[1].imshow(f2)
            plt.show()
        
        #save images
        from psnr import psnr, compressJ, save
        from PIL import Image
        save(f1, os.path.join(args.dir, "org.bmp") )
        save(f2, os.path.join(args.dir, "myJpeg.jpg") )


def run(args):
    args = parse_args(args)
    args.device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")

    model_ft, input_size = initialize_model(args,use_pretrained=True)
    image_datasets, dataloaders_dict = load_data(args, input_size) 
    
    optimizer_ft = create_optimizer(args, model_ft)
    criterion = nn.CrossEntropyLoss()
    # Train and evaluate
    model_ft, hist = train_model(args=args, model=model_ft, dataloaders=dataloaders_dict, criterion=criterion, optimizer=optimizer_ft, is_inception=(args.model_name=="inception") )
    if args.iftrain:
        torch.save(model_ft.state_dict(), os.path.join(args.dir,"model.final"))
    summarize(args, model_ft, image_datasets)
    return hist[0].cpu().numpy()

if __name__=='__main__':
    sys.exit(run(sys.argv[1:]))
