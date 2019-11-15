"""
Finetuning Torchvision Models
=============================

**Author:** `Nathan Inkawhich <https://github.com/inkawhich>`__

"""



from __future__ import print_function 
from __future__ import division
import sys,PIL,torch,os,time,copy,argparse,torchvision
from timeit import default_timer as timer
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from jpeg_layer import *
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


######################################################################
#
def parse_args(args):
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

    parser.add_argument('--data_dir', '-d', type=str,\
        # default='/data/zhijing/ImageNet/bmps/', \
        default='/data/datasets/ILSVRC2012/',\
        help='Directory of the input data. \
        String. Default: /data/zhijing/ImageNet/bmps/')
    parser.add_argument('--model_name', '-m', type=str,\
        default='squeezenet',\
        help = 'NN models to choose from [resnet, alexnet, \
        vgg, squeezenet, densenet, inception]. \
        String. Default: squeezenet')
    
    parser.add_argument('--num_classes', '-c', type=int,\
        default = 1000,\
        help = 'Number of classes in the dataset. \
        Integer. Default: 1000')
    
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

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.num_epochs//2], gamma=0.1)
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    phases =['train', 'val'] 
    if not args.iftrain:
        phases = ['val']
    for epoch in range(args.num_epochs):
        scheduler.step()
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 10)

        qt_param = getattr(model,'0').LQ.qtable
        print('Quantization Tables:', qt_param.shape)
        print('QT-Value (rounded):', torch.round(qt_param.data))

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_corrects_5 = 0

            # Iterate over data.
            for idx, (inputs, labels) in enumerate(dataloaders[phase]):
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
                    # _, preds = torch.max(outputs, 1)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        # if idx==0:
                        #     print('QT-Gradient:', qt_param.grad)
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)

                # for jj, correct_class in enumerate(labels.data.cpu().numpy()):
                #     if labels.data == preds[jj, 0]:
                #         num_top1_correct += 1
                #     if correct_class in output_index[jj, :]:
                #         num_top5_correct += 1

                top1, top5 = nCorrect(outputs, labels.data, topk=(1,5))
                # raise Exception(top1.item(), top5.item())
                running_corrects += top1.item()
                running_corrects_5 += top5.item()


            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_acc5 = running_corrects_5.double() / len(dataloaders[phase].dataset)

            print('[{}] Loss:{:.4f}; Acc1:{:.4f}%; Acc5:{:.4f}%'.format(phase, epoch_loss, epoch_acc, epoch_acc5))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        # print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def nCorrect(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            # res.append(correct_k.mul_(100.0 / batch_size))
            res.append(correct_k)
        return res

######################################################################
# Set Model Parameters’ .requires_grad attribute
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## 
def set_parameter_requires_grad(model, first, feature_extract, quant_only=False, cnn_only=False):
    if first and feature_extract:
        quant_only = True
        cnn_only = True
    for name, param in model.named_parameters():
        if  (quant_only and 'qtable' not in name) or\
            (cnn_only and 'qtable' in name):
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
        # set_parameter_requires_grad(model_ft,\
        #    True, args.feature_extract)
        # num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, args.num_classes)
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
        # load_from = os.path.join(args.dir,"model.final")
        # if args.quant_only and not args.cnn_only:
        #     model_ft.load_state_dict(torch.load(load_from))
        # if not args.iftrain:
        #     model_ft.load_state_dict(torch.load(load_from))
        model_ft = sequential(
            JpegLayer(rand_qtable=args.rand_qtable, cnn_only=args.cnn_only, quality=args.quality),
            model_ft
        )
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
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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

    params_quantize = []
    params_model = [] #model_ft.parameters()
    params_output = []
    print("Trainable Params:")
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            if 'qtable' in name:
                params_quantize.append(param)
                print('qtable\t', name)
            elif 'fc' in name:
                params_output.append(param)
                print('fc\t', name)
            else:
                params_model.append(param)
                print('net\t',name)
    if args.iftrain:
        # optimizer_ft = optim.SGD(
        #     [{'params': params_model}, {'params': params_quantize, 'lr': 0.00001} ], 
        #     lr = 0.0005, momentum=0.9)
        optimizer_ft = optim.SGD(
            [{'params': params_quantize, 'lr': 0.001} ], 
            lr = 0.001, momentum=0.9)
    else:
        optimizer_ft = None
        # optim.SGD([{'params': params_model},\
           # {'params': params_quantize, 'lr': 0.005, 'momentum':0.9}], lr=0.0005, momentum=0.9)
    return optimizer_ft

def summarize(args, model_ft, image_datasets):
    #print the trained quantization matrix
    if args.qtable:
        print('--------- the trained quantize table ---------')
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True and\
                    "qtable" in name:
                # print(name, param.data*255)
                print(name, torch.round(param.data))

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
        # print('after JpegLayer:', f2)
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

def eval_model(pt_model, data_loader, args):
    #torch.backends.cudnn.benchmark = True
    num_images = 0
    num_top1_correct = 0
    num_top5_correct = 0
    predictions = []
    start = timer()
    with torch.no_grad():
        enumerable = enumerate(data_loader)
        total = int(math.ceil(len(data_loader) / args.batch_size))
        desc = 'Batch'
        enumerable = tqdm(enumerable, total=total, desc=desc)
        for ii, (img_input, target) in enumerable:
            # if ii==50:
            #     break
            img_input = img_input.to(args.device)
            target = target.to(args.device)
            # img_input = img_input.cuda(non_blocking=True)

            if args.add_jpeg_layer:
                outputs,_,_ = pt_model(img_input)
            else:
                outputs = pt_model(img_input)
            _, output_index = outputs.topk(k=5, dim=1, largest=True, sorted=True)
            # print(output_index, target)
            output_index = output_index.cpu().numpy()
            predictions.append(output_index)
            for jj, correct_class in enumerate(target.cpu().numpy()):
                if correct_class == output_index[jj, 0]:
                    num_top1_correct += 1
                if correct_class in output_index[jj, :]:
                    num_top5_correct += 1
            num_images += len(target)
    end = timer()
    predictions = np.vstack(predictions)
    assert predictions.shape == (num_images, 5)
    top1_acc = num_top1_correct / num_images
    top5_acc = num_top5_correct / num_images
    total_time = end - start
    tqdm.write('    Evaluated {} images'.format(num_images))
    tqdm.write('    Top-1 accuracy: {:.2f}%'.format(100.0 * top1_acc))
    tqdm.write('    Top-5 accuracy: {:.2f}%'.format(100.0 * top5_acc))
    tqdm.write('    Total time: {:.1f}  (average time per image: {:.2f} ms)'.format(total_time, 1000.0 * total_time / num_images))

    return top1_acc,top5_acc


def run(args):
    args = parse_args(args)
    args.device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")

    model_ft, input_size = initialize_model(args,use_pretrained=True)
    image_datasets, dataloaders_dict = load_data(args, input_size) 

    ### evaluate initial model
    model_ft.eval()
    _, _ = eval_model(model_ft, dataloaders_dict['val'], args)
    ### Train and evaluate
    optimizer_ft = create_optimizer(args, model_ft)
    criterion = nn.CrossEntropyLoss()
    model_ft.train()
    model_ft, hist = train_model(args=args, model=model_ft, dataloaders=dataloaders_dict, criterion=criterion, optimizer=optimizer_ft, is_inception=(args.model_name=="inception") )
    if args.iftrain:
        torch.save(model_ft.state_dict(), os.path.join(args.dir,"model.final"))

    ### evaluate retrained model
    model_ft.eval()
    _, _ = eval_model(model_ft, dataloaders_dict['val'], args)
    summarize(args, model_ft, image_datasets)
    # return hist[0].cpu().numpy()

if __name__=='__main__':
    # python src/model/cnn3.py --gpu_id=3 --model_name=resnet --batch_size=32 --num_epochs=50 --quant_only
    sys.exit(run(sys.argv[1:]))
