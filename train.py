# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

#import torchvision
import torchvision.transforms as transforms

from model.models import ProposedNetwork, init_weights

from rap_loader import RAPDataset
from peta_loader import PETADataset

from sklearn.metrics import precision_score,f1_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix

# Argument parsing
# python train_adience.py -t ksize -f 3 --resume
parser = argparse.ArgumentParser(description='TGW for Age Estimation Training',
  formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--resume', '-r', action='store_true', 
  help='resume from checkpoint')
parser.add_argument('--lr', default=1e-4, type=float, 
  help='initial learning rate')
parser.add_argument('--epochs', default=200, type=int, 
  help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, 
  help='manual epoch number (useful on restarts)')
parser.add_argument('--train_batch_size', default=8, type=int, 
  help='train batch size')
parser.add_argument('--val_batch_size', default=8, type=int, 
  help='validation batch size')
parser.add_argument('--num_workers', default=0, type=int, 
  help='number of queue workers for dataloader')
parser.add_argument('--data_path', default='./data/Adience', type=str, 
  help='database path')
parser.add_argument('--checkpoint_path', default='./checkpoint', type=str, 
  help='checkpoint file path')
parser.add_argument('--fold', '-f', default='1', help='0/1/2/3/4')     
parser.add_argument('--gpu', default='7', type=str, help='GPU Number')
parser.add_argument('--type', '-t', default='ksize',
  help='ksize: learns sampling grid (Best Method, abbreviation=Kernel SIZE) \
    \n all: learns sampling grids and parameters of Gabor wavelets \
    \n ~ksize: learns other parameters of Gabor wavelets except orientation \
    \n none: do not learn any parameter of Gabor wavelets w/ steering.')
args = parser.parse_args()

# Set Training GPU Number
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
   
# Check if cuda is available
USE_CUDA = torch.cuda.is_available()

DATASET = "PETA"

if DATASET == "PETA":
    ATTRIBS = 35
    pckl_location = ".pckl";
    pckl_location = "peta_data_144x48_35attrs.pckl";   
    dataset_class = PETADataset
else:
    ATTRIBS = 51;
    dataset_class = RAPDataset
    pckl_location = 'rap_data_rgb_144x48_51attrs.pckl';

def train(train_loader, net, criterion, optimizer):
  """
  Training for one epoch. Accuracy is given by %

  Args: 
    train_loader: instances of DataLoader with Adience DB
    net: instance of models.ProposedNetwork
    criterion: loss for network (usually cross-entropy)
    optimizer: weight optimizer for network
  """

  net.train()
  train_loss = 0
  correct = 0    
  total = 0
  all_targets = []
  all_outputs = []
  print('training....')
  for batch_idx, (in1, in2, in3, targets) in enumerate(train_loader):

    # for batch_normalization
    if (in1.size()[0] < 2):
      continue

    if USE_CUDA:
        in1, in2, in3, targets = in1.cuda(), in2.cuda(), in3.cuda() , targets.cuda()

    in1, in2, in3, targets = Variable(in1), Variable(in2), Variable(in3), Variable(targets.squeeze())
    
    optimizer.zero_grad()
    
    outputs = net(in1, in2, in3)

    if (torch.sum((outputs.data != outputs.data))):
      print('NaN error')
      return
    
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    
    outputs = outputs.data > 0.5
    outputs = outputs.to(torch.float32)
    
    all_targets.append(targets.flatten().to("cpu").to(torch.int).numpy())
    all_outputs.append(outputs.flatten().to("cpu").to(torch.int).numpy())

    train_loss += loss.item()
    
    total += targets.size(0)*ATTRIBS        
    correct += (outputs==targets).sum();
    
          
  all_outputs = np.concatenate(all_outputs)
  all_targets = np.concatenate(all_targets)
  
  precis = precision_score( all_outputs, all_targets, average='weighted')
  recall = recall_score( all_outputs, all_targets, average='weighted') 
  f1score = f1_score( all_outputs, all_targets, average='weighted')  
  acc_score = 100.*( all_outputs==all_targets ).sum()/total;
  
  print('---> Training Epoch::: F1-score: %.3f | Recall: %.3f | | Precision: %.3f | Accuracy: %.3f' % (100.*f1score, 100.*recall, 100.*precis, acc_score))

def validate(val_loader, net, criterion):
  """
  Validation for one epoch. Accuracy is given by %

  Args: 
    val_loader: instances of DataLoader with Adience DB
    net: instance of models.ProposedNetwork
    criterion: loss for network (usually cross-entropy)
  """
  net.eval()
  val_loss = 0
  correct = 0
  total = 0
  all_targets = []
  all_outputs = []
  print('validating ....')
  for batch_idx, (in1, in2, in3, targets) in enumerate(val_loader):

    if USE_CUDA:
      in1, in2, in3, targets = in1.cuda(), in2.cuda(), in3.cuda() , targets.cuda() 

    in1 = Variable(in1, volatile=True)
    in2 = Variable(in2, volatile=True)
    in3 = Variable(in3, volatile=True)
    targets = Variable(targets.squeeze())
    outputs = net(in1, in2, in3)
    
    if outputs.dim() < 2:
        outputs.unsqueeze(0)        

    loss = criterion(outputs, targets)
    outputs = outputs.data > 0.5
    outputs = outputs.to(torch.float32)   
       
    all_targets.append(targets.flatten().to("cpu").to(torch.int).numpy())
    all_outputs.append(outputs.flatten().to("cpu").to(torch.int).numpy())

    val_loss += loss.item()#data[0]

    total += targets.size(0)*ATTRIBS
    
    correct += (outputs==targets).sum();
    
  all_outputs = np.concatenate(all_outputs)
  all_targets = np.concatenate(all_targets)  
  cur_accuracy = 100.*(np.array(all_outputs)==np.array(all_targets) ).sum()/total
  recal = recall_score(np.array(all_outputs), np.array(all_targets), average='weighted')
  precis = precision_score(np.array(all_outputs), np.array(all_targets), average='weighted')  
  f1score = f1_score(np.array(all_outputs), np.array(all_targets), average='weighted')  
  print('---> Validation: F1-score: %.3f | Recall: %.3f | | Precision: %.3f | Accuracy: %.3f' % (100.*f1score, 100.*recal, 100.*precis, cur_accuracy))  
   
  return cur_accuracy

def main():
   
  best_accuracy = 0 
  checkpoint_str = 'ProposedNetwork_{}_fold{}'.format(args.type, args.fold)    

  # Create proposed network
  print('# Creating proposed network')

  net = ProposedNetwork(args.type)
  
  if USE_CUDA:
      net = torch.nn.DataParallel(net).cuda() 
  
  if args.resume:
    print('#- Resume model from checkpoint')
    assert os.path.isdir('checkpoint'), \
      '#- Error: no checkpoint directory found!'

    checkpoint = torch.load(os.path.join(args.checkpoint_path, 
      checkpoint_str + '.checkpoint'))

    best_accuracy = checkpoint['acc']
    args.start_epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['net'])
  
  else:
    print('#- Initilize weights in proposed network')
    net.apply(init_weights)

  if USE_CUDA:
    net.cuda()
    cudnn.benchmark = True

  # Criterion, Optimizer  
  print('# Creating loss function and optimizer')

   # Training and validation data Preparation
  print('# Preparing trainning and validation data')
  
  transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
      
  transform_val = transforms.Compose([    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

  train_set = dataset_class(pkl_filename=pckl_location, 
    transforms = transform_train, reqtype='train')  # Give image_dirs as list format
  
  train_loader = torch.utils.data.DataLoader(train_set, 
                    batch_size=args.train_batch_size, 
                    shuffle=True, 
                    num_workers=args.num_workers)
  
  val_set = dataset_class(pkl_filename=pckl_location, 
    transforms = transform_val, reqtype='test') # Give image_dirs as list format
  
  val_loader = torch.utils.data.DataLoader(val_set, 
                batch_size=args.val_batch_size, 
                shuffle=False, 
                num_workers=args.num_workers)
        
  criterion = nn.BCEWithLogitsLoss( )
  optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
  
  if args.resume:
    optimizer.load_state_dict(checkpoint['optimizer'])  
  
  # Training loop
  for epoch in range(args.start_epoch, args.epochs):
    print('\nEpoch:{}, Type:{}, Fold:{}'.format(epoch, args.type, args.fold))

    # Learning rate decay for every 30 epoches
    adjust_learning_rate(optimizer, epoch)

    # Training and validation for one epoch
    train(train_loader, net, criterion, optimizer)    
    cur_accuracy = validate(val_loader, net, criterion)

    # Remember best accuracy model and save checkpoint
    is_best = cur_accuracy > best_accuracy
    best_accuracy = max(cur_accuracy, best_accuracy)
    save_checkpoint({
      'net': net.state_dict(),
      'acc': best_accuracy,
      'epoch': epoch + 1,
      'optimizer' : optimizer.state_dict(),
    }, is_best, checkpoint_str)

    print(' best f1score = {}'.format(best_accuracy))     

def adjust_learning_rate(optimizer, epoch):
  """ Initial learning rate decayed by 10 every 30 epochs """

  lr = args.lr * (0.1 ** (epoch // 15))
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
    
def save_checkpoint(state, is_best, checkpoint_str):
  """ Save checkpoint for each epoch and best accuracy model """

  print(' save checkpoint')
  torch.save(state, 
    os.path.join(args.checkpoint_path, checkpoint_str + '.checkpoint'))

  if is_best:
    print(' save best accurary model')

    shutil.copyfile(
      os.path.join(args.checkpoint_path, checkpoint_str + '.checkpoint'), 
      os.path.join(args.checkpoint_path, checkpoint_str + '.best_accuracy'))

if __name__ == '__main__':
  main()