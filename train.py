import os
import sys
import time
import argparse
import torch
import numpy as np
import torchvision
from PIL import Image
from torchvision import transforms

## return True value
def get_true():
    return(True)

# directory initial function
def dir_init(dir):
    if not os.path.exists(dir):
        try:
            os.mkdir(dir)

        except Except as e:
            sys.stderr.write('[Error][%s]' % (e))
            sys.stderr.flush()
            return(False)

    return(True)

## train func
def build_model(data_dir, arch, hidden_units):
    ## load vgg19
    # Build and train your network
    # model = torchvision.models.vgg19_bn(pretrained = True)
    model = getattr(torchvision.models, arch)
    model = model(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False

    dropout_rate = 0.5
    classifier = torch.nn.Sequential(torch.nn.Linear(25088, hidden_units),
                                torch.nn.ReLU(),
                                torch.nn.Dropout(p = dropout_rate),
                                torch.nn.Linear(hidden_units, 102))
    model.classifier = classifier
    return(model, classifier)

## train func
def train(model, train_loader, valid_loader, learning_rate, gpu, epochs):
    if gpu:
        device = 'cuda'

    else:
        device = 'cpu'

    learning_rate = learning_rate
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr = learning_rate)
    model.to(device)
    epoch = epochs
    # print('debug {}'.format(epochs))
    # print('debug type {}'.format(type(epochs)))

    ## start epoch
    for e in range(epoch):
        trunning_loss = 0
        trunning_corrects = 0
        vrunning_loss = 0
        vrunning_corrects = 0
    
        # train
        for ii, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            start = time.time()
            outputs = model.forward(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            trunning_loss += loss.data[0]
            trunning_corrects += torch.sum(preds == labels.data)
        
        base = len(train_loader)
        print('[train][Epoch {}]Loss: {:.6f} Acc: {:.6f}'.format(e, trunning_loss / base, trunning_corrects.double() / (base * 64)))
        print('[train][Time per batch: {:.6f} seconds]'.format((time.time() - start) / base))
    
        # valid
        for ii, (inputs, labels) in enumerate(valid_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            start = time.time()
            outputs = model.forward(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            vrunning_loss += loss.data[0]
            vrunning_corrects += torch.sum(preds == labels.data)
        
        base = len(valid_loader)
        print('[valid][Epoch {}]Loss: {:.6f} Acc: {:.6f}]'.format(e, vrunning_loss / base, vrunning_corrects.double() / (base * 64)))
        print('[valid][Total Time : {:.6f} seconds]'.format(time.time() - start))

    return(optimizer)
    
## train func
def test(model, test_loader, gpu):
    if gpu:
        device = 'cuda'

    else:
        device = 'cpu'

    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)

    # test
    trunning_loss = 0
    trunning_corrects = 0
    base = len(test_loader)
    for ii, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        start = time.time()
        outputs = model.forward(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        trunning_loss += loss.data[0]
        trunning_corrects += torch.sum(preds == labels.data)
    
    print('[test][Loss: {:.6f} Acc: {:.6f}]'.format(trunning_loss / base, trunning_corrects.double() / (base * 64)))
    print('[test][Total Time : {:.6f} seconds]'.format(time.time() - start))

## save model
def save(save_dir, model, arch, train_dataset):
    checkpoint = {'arch': arch,
        'classifier' : model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': train_dataset.class_to_idx}
    torch.save(checkpoint, '{}checkpoint.pth'.format(save_dir))

## main run part
if __name__ == '__main__':
    ## read par
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help = 'Training Set Data Dir', default = './flowers/')
    parser.add_argument('--save_dir', help = 'Set directory to save checkpoints', default = '')
    parser.add_argument('--arch', help = 'Arch can be choose from \"vgg19|vgg19_bn|vgg16|vgg16_bn\" or not set it', default = 'vgg19_bn')
    parser.add_argument('--learning_rate', help = 'Learning Rate', default = 0.001, type = float)
    parser.add_argument('--hidden_units', help = 'Hidden Units', default = 1024, type = int)
    parser.add_argument('--epochs', help = 'Epoches', default = 10, type = int)
    parser.add_argument('--gpu', help = 'To use GPU.', action = 'store_true', default = False)
    args = parser.parse_args()

    ## check model input
    support_lst = ['vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']
    if (args.arch not in ['vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']):
        print('Only Support {}'.format(support_lst))
        sys.exit(-1)

    ## precheck save dir
    if (args.save_dir != '') and dir_init(args.save_dir):
        print('Could not create dir')
        sys.exit(-2)

    ## debug print
    # print(args.data_dir)
    # print(args.save_dir)
    # print(args.arch)
    # print(args.learning_rate)
    # print(args.hidden_units)
    # print(args.epochs)
    # print(args.gpu)

    ## pre
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    batch_size = 64
    
    ## define model
    # Define your transforms for the training, validation, and testing sets
    train_transforms = torchvision.transforms.Compose([transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    valid_transforms = torchvision.transforms.Compose([transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_transforms = torchvision.transforms.Compose([transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_dataset = torchvision.datasets.ImageFolder(root = train_dir, transform = train_transforms)
    valid_dataset = torchvision.datasets.ImageFolder(root = valid_dir, transform = valid_transforms)
    test_dataset = torchvision.datasets.ImageFolder(root = test_dir, transform = test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(dataset = valid_dataset, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)


    ## build model
    model, classifier = build_model(args.data_dir, args.arch,args.hidden_units)

    ## train model
    optimizer = train(model, train_loader, valid_loader, args.learning_rate, args.gpu, args.epochs)

    ## test on TestSet
    test(model, test_loader, args.gpu)

    ## save model
    save(args.save_dir, model, args.arch,train_dataset)
