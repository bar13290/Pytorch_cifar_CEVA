'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np

from pytorch_modelsize import SizeEstimator
from models import *
#from utils import progress_bar

flag_quan = 0
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--test', '-t', action='store_true',
                    help='Using the model for testing')
parser.add_argument('--quan', '-q', action='store_true',
                    help='Quantize the model, check and save')
parser.add_argument('--onnx', '-o', action='store_true',
                    help='Save the model as onnx file')
args = parser.parse_args()


path_to_model = './checkpoint/ckpt_v2.pth'
path_to_quantize_model = './checkpoint/ckpt_v2_quantize.pth'
path_to_onnx = './onnx/v1.onnx'


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 1  # start from epoch 0 or last checkpoint epoch

# Data

# number of subprocesses to use for data loading
num_workers = 2
# how many samples per batch to load
train_batch_size = 128
test_batch_size = 100
# percentage of training set to use as validation
valid_size = 0.1

print('==> Preparing data..')

# convert data to a normalized torch.FloatTensor
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# choose the training and test datasets
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

# obtain training indices that will be used for validation
num_train = len(trainset)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size = train_batch_size, sampler=train_sampler, num_workers=num_workers)

validloader = torch.utils.data.DataLoader(
    trainset, batch_size = train_batch_size, sampler=valid_sampler, num_workers=num_workers)

testloader = torch.utils.data.DataLoader(
    testset, batch_size = test_batch_size, shuffle=False, num_workers=num_workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
#net = ResNet18_relu6()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
#net = SimpleDLA()


net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(path_to_model)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def cal(net_prepared):
    net_prepared.eval()
    test_loss = 0
    correct = 0
    total = 0
    print("Using the VALIDATION data set for calibration")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            print("Cal Batch number: {}".format(batch_idx))
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net_prepared(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(validloader), 'Epoch: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (epoch, test_loss/(batch_idx+1), 100.*correct/total, correct, total))



def quantize(path_to_wanted_model):
    #Loading the post training parameters
    print('==> Loading the model parameters..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    load_model = torch.load(path_to_wanted_model)
    net.load_state_dict(load_model['net'])
    net.eval()
    # attach a global qconfig, which contains information about what kind of observers to attach
    net.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.backends.quantized.engine = 'fbgemm'
    # Fuse the activations to preceding layers
    net_fused = torch.quantization.fuse_modules(net, ['conv1', 'bn1'], ['conv2', 'bn2'])
    # Prepare the model for static quantization
    net_prepared = torch.quantization.prepare(net_fused, inplace=False)
    #calibrate the prepared model to determine quantization parameters for activations
    cal(net_prepared)
    net_int8 = torch.quantization.convert(net_prepared, inplace=False)
    #Saving the model
    print("saving the quantized model")
    state = {
        'net': net_int8.state_dict(),
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, path_to_quantize_model)
    print("Fisinshed quantizing he model")
    #Check performances
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    net_int8.eval()
    # iterate over test data
    for data, target in testloader:
        print(target)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = net_int8.eval_quant(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update test loss
        test_loss += loss.item() * data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy())
        # calculate test accuracy for each object class
        for i in range(test_batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # average test loss
    test_loss = test_loss / len(testloader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(10):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))

    return

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    total_param = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total trainable parameters : {}".format(total_param))
    train_loss = 0
    correct = 0
    total = 0
    print("TRAINING DATA SET")
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Epoch: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (epoch, train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return (train_loss/(batch_idx+1) ,(100.*correct/total))

def val(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    print("VALIDATION DATA SET")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(validloader), 'Epoch: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (epoch, test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('\nSaving Best Model (epoch : {}) to {}..'.format(epoch, path_to_model))
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, path_to_model)
        best_acc = acc
    return test_loss / (batch_idx + 1), (100. * correct / total)

def test(path_to_model):
    # Load checkpoint.
    print('==> Loading the model parameters..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    load_model = torch.load(path_to_model)
    net.load_state_dict(load_model['net'])
    # track test loss
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    net.eval()
    # iterate over test data
    for data, target in testloader:
        print(target)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = net(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update test loss
        test_loss += loss.item() * data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy())
        # calculate test accuracy for each object class
        for i in range(test_batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # average test loss
    test_loss = test_loss / len(testloader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(10):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))

def onnx(onnx_name):
    print('==> Loading the model parameters..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    load_model = torch.load(path_to_model)
    net.load_state_dict(load_model['net'])
    dummy_input = torch.randn(10, 3, 32, 32)
    if not os.path.isdir('onnx'):
        os.mkdir('onnx')
    print("Starting export onnx file!")
    torch.onnx.export(net, dummy_input, onnx_name, verbose=False)
    print("Saved onnx file at : {}".format(onnx_name))
    return


if (args.onnx):
    onnx(path_to_onnx)

if (args.test):
    print("Start Testing")
    if args.quan:
        test(path_to_quantize_model)
    else:
        test(path_to_model)
    print("Finish Testing")

elif (args.quan or flag_quan==1):
    print("Quantizing model: {}".format(path_to_model))
    quantize(path_to_model)

else:
    print("#####  Starting The Training Session!!!  #####")
    arr_loss_train = []
    arr_loss_val = []
    arr_acc_train = []
    arr_acc_val = []
    for epoch in range(start_epoch, start_epoch+3):
        loss_train, acc_train = train(epoch)
        loss_val, acc_val = val(epoch)
        scheduler.step()

        # Create accuracy and loss Plots
        arr_loss_train.append(loss_train)
        arr_loss_val.append(loss_val)
        arr_acc_train.append(acc_train)
        arr_acc_val.append(acc_val)

        # Plot the loss dev Vs train
        plt.figure()
        plt.plot(np.arange(1, epoch + 1), arr_loss_train, np.arange(1, epoch + 1), arr_loss_val)
        plt.title("Accuracy - Dev | Relu6 model")
        plt.xlabel("Epoch")
        plt.xticks(range(1, epoch + 1))
        plt.ylabel("Accuracy (%)")
        plt.grid(True)
        plt.legend(['Training Loss', 'Dev Loss'])
        if not os.path.isdir('plots'):
            os.mkdir('plots')
        print("Saving Fig...")
        plt.savefig('./plots/accuracy_vs_epochs_v2.png')
    print("#####  Finished The Training Session!!!  #####")








