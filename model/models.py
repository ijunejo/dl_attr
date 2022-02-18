# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module

# Maximum size of sampling grid (for clipping of zeta in equation (2))
MAX_KERNEL_SIZE = 12

# Number of Gabor wavelets will be used as basis for streering
DIM_GABOR_BASIS = 10
M_PI = 3.14159265358979323846

class GaborKernelFunction(Function):

  """
  Implementation of equation (1) to (4) in paper.
  It generate Gabor wavelet and calculate its gradient w.r.t parameters
  Number of orientations in generation of battery of Gabor wavelets 
    is defined as in Section 2.2: 10 equally spaced orientations with offset 9'
    (9', 27', 45', 63', 81', 99', 117', 135', 153', 171')
  Therefore the shape of battery of Gabor wavelet is
    [10, 1, height of input feature, width of input feature]


  Args:
    half_ksize: zeta/2 in equation (2)
    sig, gamma, lambd: parameters of Gabor wavelets 
      (will be given by GaborKernelParamModule)
    type: operating mode of proposed network
  """

  @staticmethod
  def forward(ctx, half_ksize, sig, gamma, lambd, type):
    
    ctx.type = type 
    
    sig = sig.cuda();
    gamma = gamma.cuda()
    lambd.cuda()  
    
    # sampling grid = 
    #   [-(2 * rounded_half_ksize + 1), 2 * rounded_half_ksize + 1]
    rounded_half_ksize = \
      int(torch.clamp(torch.round(half_ksize), min=1, max=MAX_KERNEL_SIZE))
    
    # Cuda memory allocation for Gabor kernels
    gabor_kernels = torch.zeros(DIM_GABOR_BASIS, 
      2 * rounded_half_ksize + 1, 
      2 * rounded_half_ksize + 1).cuda() 
    
    ksize = int(2 * rounded_half_ksize + 1);  
    y, x = torch.meshgrid(
            [
                torch.linspace(-rounded_half_ksize + 1, rounded_half_ksize + 0, ksize),
                torch.linspace(-rounded_half_ksize + 1, rounded_half_ksize + 0, ksize),
            ])

    theta = torch.tensor([0.1571, 0.4712, 0.8029, 1.0996, 1.4137, 1.7279, 2.0420, 2.3562, 2.6704,2.9845]);
    theta = theta.view(-1,1,1);
    
    X = x * torch.cos(theta) + y * torch.sin(theta)
    Y = -x * torch.sin(theta) + y * torch.cos(theta)
    
    X = X.cuda()
    Y = Y.cuda()
    
    gabor_kernels = torch.exp( -( X**2 + gamma**2 * Y**2) * 0.5 * sig**-2) * torch.cos( 2 * M_PI * X * lambd**-1 );        

    # Change shape of Gabor wavelets into `Channel x Width x Height' format
    #   to use with nn.conv2d
    gabor_kernels = gabor_kernels.unsqueeze(1)

    # Save variables for backprop
    ctx.save_for_backward(sig, gamma, lambd, half_ksize, gabor_kernels)

    return gabor_kernels

  @staticmethod
  def backward(ctx, grad_output):

    sig, gamma, lambd, half_ksize, gabor_kernels = ctx.saved_tensors
    
    sig = sig.cuda();
    gamma = gamma.cuda()
    lambd = lambd.cuda()
    gabor_kernels = gabor_kernels.cuda()

    # Change shape of tensors as
    # `DIM_GABOR_BASIS x Channel(=1) of Input x Width x Height' into
    # `DIM_GABOR_BASIS x Width x Height'
    gabor_kernels = gabor_kernels.squeeze(1)
    grad_output = grad_output.squeeze(1)

    # Cuda memory allocation for gradient calculated per batch
    grad_sig   = torch.zeros(grad_output.size()).cuda()
    grad_gamma = torch.zeros(grad_output.size()).cuda() 
    grad_lambd = torch.zeros(grad_output.size()).cuda() 
    grad_ksize = torch.zeros(grad_output.size()).cuda() 

    rounded_half_ksize = \
      int(torch.clamp(torch.round(half_ksize), min=1, max=MAX_KERNEL_SIZE))
       
    ksize = 2 * rounded_half_ksize + 1; 

    #Calculate gradients for backprop
    y, x = torch.meshgrid(
            [
                torch.linspace(-rounded_half_ksize + 1, rounded_half_ksize + 0, ksize),
                torch.linspace(-rounded_half_ksize + 1, rounded_half_ksize + 0, ksize),
            ])

    theta = torch.tensor([0.1571, 0.4712, 0.8029, 1.0996, 1.4137, 1.7279, 2.0420, 2.3562, 2.6704,2.9845]);
    theta = theta.view(-1,1,1);
    X = x * torch.cos(theta) + y * torch.sin(theta)
    Y = -x * torch.sin(theta) + y * torch.cos(theta)
    
    theta = theta.cuda()
    half_ksize = half_ksize.cuda()
    X = X.cuda()
    Y = Y.cuda()
    x = x.cuda()
    y = y.cuda()
    
    dkern_by_Y = - (gamma**2 * sig**-2 * Y);
    dkern_by_X = - (X * sig**-2) - torch.tan(2 * M_PI * lambd**-1 * X) * 2 * M_PI * lambd**-1;
    
    dkern_by_Y = dkern_by_Y.cuda()
    dkern_by_X = dkern_by_X.cuda()
    
    grad_ksize_element = (dkern_by_X * torch.cos(theta) + dkern_by_Y * (-torch.sin(theta))) * x / (half_ksize);
    grad_ksize_element = grad_ksize_element + (dkern_by_X * torch.sin(theta) + dkern_by_Y * torch.cos(theta)) * y / (half_ksize);
    
    grad_gamma_element = -Y**2 * gamma * sig**-2;
    grad_lambd_element = torch.tan(2 * M_PI * X * lambd**-1) * (2 * M_PI * X * lambd**-2);   
    grad_sig_element = (X**2 +  (gamma**2 * Y**2)) * sig**-3;                                         
    
    grad_sig_element = gabor_kernels * grad_sig_element;
    grad_ksize_element = gabor_kernels * grad_ksize_element;
    grad_gamma_element = gabor_kernels * grad_gamma_element;
    grad_lambd_element = gabor_kernels * grad_lambd_element;
    
    grad_sig   = grad_output * grad_sig_element;
    grad_ksize = grad_output * grad_ksize_element;  
    grad_lambd = grad_output * grad_lambd_element;
    grad_gamma = grad_output * grad_gamma_element;
    
    # Calculate gradient of each parameter 
    #   by summing over results from each batches
    if ctx.type == 'all':
      grad_sig   = Variable(grad_sig  ).sum()
      grad_gamma = Variable(grad_gamma).sum()
      grad_lambd = Variable(grad_lambd).sum()
      grad_ksize = Variable(grad_ksize).sum()
    
    if ctx.type == 'ksize':
      grad_ksize = Variable(grad_ksize).sum()
      grad_sig   = None
      grad_gamma = None
      grad_lambd = None
    
    if ctx.type == '~ksize':
      grad_ksize = None
      grad_sig   = Variable(grad_sig  ).sum()
      grad_gamma = Variable(grad_gamma).sum()
      grad_lambd = Variable(grad_lambd).sum()
    
    if ctx.type == 'none':
      grad_ksize = None
      grad_sig   = None
      grad_gamma = None
      grad_lambd = None

    return grad_sig, grad_gamma, grad_lambd, grad_ksize, None

class GaborKernelParamBlock(Module):

  def __init__(self, num_in_ftrs, num_out_ftrs, kernel_size, maxpool_size):
    """
    Building block of contructing GaborKernelParamModule
    Consists of convolution layer with maxpool
    Batchnorm is added for each layer

    Args:
      num_in_ftrs: number of channels in input features
      num_out_ftrs: the number of channels of features 
        will be generated by convolution layer.
      kernel_size: size of convolution 
      maxpool_size: size of maxpool
    """
    super(GaborKernelParamBlock, self).__init__()

    self.conv = nn.Conv2d(num_in_ftrs, num_out_ftrs, kernel_size=kernel_size)
    self.max = nn.MaxPool2d(2, stride=2)
    self.bn = nn.BatchNorm2d(num_out_ftrs)

  def forward(self, input):

    output = self.bn(self.max(F.relu(self.conv(input))))
    return output

class GaborKernelParamModule(Module):

  def __init__(self, input_size, num_in_ftrs, preset, type):
    """
    Estimation of parameters of Gabor wavelets for each operating mode
    We used same conv features and different FC layer for each parameter
    Parameters that do not need training set as values in preset.
    CNN used for parameter estimation has variable depth 
      according to the size of input

    Args: 
      input_size: size of input features
      num_in_ftrs: the number of channels in input features
      preset: gamma_0, lambda_0, zeta_0, sigma_0 in Equation (5)
      type: operating mode of proposed network
    """
    super(GaborKernelParamModule, self).__init__()

    self.preset = preset
    self.type = type
    
    layers = []

    # Size of convolution layer and maxpool for CNN in parameter estimation
    param_block_conv_size = 3
    param_block_maxpool_size = 2

    # Contruct CNN with output size 2x2
    size_after_block = np.floor(
      (input_size - param_block_conv_size + 1) / param_block_maxpool_size)

    while (size_after_block >= 2):
      # Set 90 as output feature # for input feature channel lesser than 90
      # Set 256 as output feature # for inpout feature channel more than 256

      if (num_in_ftrs < 90): 
        num_out_ftrs = 90
      else: 
        if (2 * num_in_ftrs > 256): 
          num_out_ftrs = 256
        else:
          num_out_ftrs = 2 * num_in_ftrs

      layers.append(GaborKernelParamBlock(num_in_ftrs, num_out_ftrs, 3, 2))
      num_in_ftrs = num_out_ftrs
      size_after_block = \
        np.floor((size_after_block - param_block_conv_size + 1) 
          / param_block_maxpool_size)

    # Make the shape of output as `batch x num_out_ftrs x 1 x 1'
    if (size_after_block == 1):
       layers.append(GaborKernelParamBlock(num_in_ftrs, num_out_ftrs, 3, 2))
    else:
      layers.append(GaborKernelParamBlock(num_in_ftrs, num_out_ftrs, 1, 1))

    # Build CNN will be used for parameter esimation
    self.conv_layer = nn.Sequential(*layers)

    # FC layer for estimation of each parameter
    if self.type == 'all':
      self.lin_ksize    = nn.Linear(256, 1)
      self.lin_ksize_drop = nn.Dropout2d(p=0.5)
      self.lin_sig    = nn.Linear(256, 1)
      self.lin_sig_drop   = nn.Dropout2d(p=0.5)
      self.lin_gamma    = nn.Linear(256, 1)
      self.lin_gamma_drop = nn.Dropout2d(p=0.5)
      self.lin_lambd    = nn.Linear(256, 1)
      self.lin_lambd_drop = nn.Dropout2d(p=0.5)

    if self.type == 'ksize':
      self.lin_ksize    = nn.Linear(256, 1)
      self.lin_ksize_drop = nn.Dropout2d(p=0.5)

    if self.type == '~ksize':
      self.lin_sig    = nn.Linear(256, 1)
      self.lin_sig_drop   = nn.Dropout2d(p=0.5)
      self.lin_gamma    = nn.Linear(256, 1)
      self.lin_gamma_drop = nn.Dropout2d(p=0.5)
      self.lin_lambd    = nn.Linear(256, 1)
      self.lin_lambd_drop = nn.Dropout2d(p=0.5)

  def forward(self, input):

    # Generate convolutional features from CNN
    param_conv_feature = self.conv_layer(input).view(-1,256)
    
    # Apply FC layers for each parameters required to be trained in
    #   given operating mode
    if self.type == 'all':
      ksize = self.lin_ksize_drop(F.relu(self.lin_ksize(param_conv_feature)))
      ksize = self.preset['ksize'] * F.sigmoid(ksize)
      sig = self.lin_sig_drop(F.relu(self.lin_sig(param_conv_feature)))
      sig = self.preset['sig'] * F.sigmoid(sig)
      gamma = self.lin_gamma_drop(F.relu(self.lin_gamma(param_conv_feature)))
      gamma = self.preset['gamma'] * F.sigmoid(gamma)   
      lambd = self.lin_lambd_drop(F.relu(self.lin_lambd(param_conv_feature)))
      lambd = self.preset['lambd'] * F.sigmoid(lambd)

    if self.type == 'ksize':
      ksize = self.lin_ksize_drop(F.relu(self.lin_ksize(param_conv_feature)))
      ksize = self.preset['ksize'] * F.sigmoid(ksize)
      sig = Variable(torch.from_numpy(np.array(
          [self.preset['sig']])).float().cuda() )
      gamma = Variable(torch.from_numpy(np.array(
          [self.preset['gamma']])).float().cuda() )
      lambd = Variable(torch.from_numpy(np.array(
          [self.preset['lambd']])).float().cuda() )
      
    if self.type == '~ksize':
      ksize = Variable(torch.from_numpy(np.array(
        [self.preset['ksize']])).float().cuda() )
      sig = self.lin_sig_drop(F.relu(self.lin_sig(param_conv_feature)))
      sig = self.preset['sig'] * F.sigmoid(sig)
      gamma = self.lin_gamma_drop(F.relu(self.lin_gamma(param_conv_feature)))
      gamma = self.preset['gamma'] * F.sigmoid(gamma)   
      lambd = self.lin_lambd_drop(F.relu(self.lin_lambd(param_conv_feature)))
      lambd = self.preset['lambd'] * F.sigmoid(lambd)

    if self.type == 'none':
      ksize = Variable(torch.from_numpy(np.array(
          [self.preset['ksize']])).float().cuda() )
      sig = Variable(torch.from_numpy(np.array(
          [self.preset['sig']])).float().cuda() )
      gamma = Variable(torch.from_numpy(np.array(
          [self.preset['gamma']])).float().cuda() )
      lambd = Variable(torch.from_numpy(np.array(
          [self.preset['lambd']])).float().cuda() )
    
    return (ksize, sig, gamma, lambd)

class GaborFeatureModule(Module):

  def __init__(self, input_size, num_in_ftrs, num_out_ftrs, preset, type):
    """ 
    Implementation of TGW
    
    Args:
      input_size: size of input features
      num_in_ftrs: the number of channels in input features
      num_out_ftrs: the number of channels of features 
        generated by Gabor wavelet.
        the number of output features is managed by channels of 1x1 convolution
        used in steering
      preset: gamma_0, lambda_0, zeta_0, sigma_0 in Equation (5)
      type: operating mode of proposed network
    """
    super(GaborFeatureModule, self).__init__()

    self.type = type
    self.param_nn = \
      GaborKernelParamModule(input_size, num_in_ftrs, preset, self.type)
    self.steering_conv = nn.Conv2d(DIM_GABOR_BASIS, num_out_ftrs,  1)

  def forward(self, input):

    # Result of Gabor filtering of input
    gabor_kernel_features = None

    # Generate parameters of Gabor wavelets through CNN
    ksize, sig, gamma, lambd = self.param_nn(input)

    # Size of batch
    batch_size = input.size()[0]

    # 1 Channel conversion
    pseudo_gray_input = torch.sum(input,1).unsqueeze(1)

    for j in range(batch_size):

      gabor_kernel_features_per_batch = None
      
      if self.type == 'all':
      
        ksize_per_batch = ksize[j, :]
        sig_per_batch   = sig[j, :]
        gamma_per_batch = gamma[j, :]
        lambd_per_batch = lambd[j, :]
      
      if self.type == 'ksize':

        ksize_per_batch = ksize[j, :]
        sig_per_batch   = sig
        gamma_per_batch = gamma
        lambd_per_batch = lambd
        
      if self.type == '~ksize':
      
        ksize_per_batch = ksize
        sig_per_batch   = sig[j, :]
        gamma_per_batch = gamma[j, :]
        lambd_per_batch = lambd[j, :]

      if self.type == 'none':
      
        ksize_per_batch = ksize
        sig_per_batch   = sig
        gamma_per_batch = gamma
        lambd_per_batch = lambd
      
      # Calculate sampling grid size
      rounded_half_ksize = \
        int(torch.clamp(torch.round(ksize_per_batch), 
          min=1, 
          max=MAX_KERNEL_SIZE))

      # Generate Gabor wavelets per batch
      self.gabor_kernel = \
        GaborKernelFunction.apply(ksize_per_batch, 
          sig_per_batch, 
          gamma_per_batch, 
          lambd_per_batch, 
          self.type)

      # Gabor filtering per batch
      # padding=rounded_half_ksize is added 
      #   to make result of each batch have same size for concatenation.
      gabor_kernel_features_per_batch = \
        F.conv2d(pseudo_gray_input[j,:].unsqueeze(0), 
          self.gabor_kernel, 
          padding = rounded_half_ksize)
      
      # Concatenation of results calculated per batch into one feature matrix
      if (gabor_kernel_features is None):
        gabor_kernel_features = gabor_kernel_features_per_batch
      else:
        gabor_kernel_features = \
          torch.cat([gabor_kernel_features, 
            gabor_kernel_features_per_batch], 0)

    # Steering of gabor features by 1x1 convolution
    output = self.steering_conv(gabor_kernel_features)

    return output

class GaborBlockModuleDagn(Module):

  def __init__(self, input_size, pool_size, num_in_ftrs, 
    num_conv_ftrs, num_gabor_ftrs, preset, type, is_last):
    """ 
    Construct 1st to 6th block in the proposed network (in Fig. 3)
    
    Args:
      input_size: size of input features
      pool_size: size of max pooling will be applied to output
      num_in_ftrs: number of channels in input features
      num_conv_ftrs: number of channels of features generated by convolution
      num_gabor_ftrs: number of channels of features generated by Gabor wavelet
      preset: gamma_0, lambda_0, zeta_0, sigma_0 in Equation (5)
      type: operating mode of proposed network
      is_last: whether block under contruction is at the front of FC layers
    """
    super(GaborBlockModuleDagn, self).__init__()

    # Definitions for TGW
    self.input_size   = input_size
    self.pool_size    = pool_size
    self.num_in_ftrs  = num_in_ftrs
    self.num_conv_ftrs  = num_conv_ftrs
    self.num_gabor_ftrs = num_gabor_ftrs
    self.preset     = preset
    self.bn_gabor = nn.BatchNorm2d(self.num_gabor_ftrs)
    self.gabor_feature_module = \
      GaborFeatureModule(self.input_size, 
        self.num_in_ftrs, 
        self.num_gabor_ftrs, 
        self.preset, 
        type)
    self.is_last = is_last
    
    # Definitions for 3x3 convolution
    self.conv = nn.Conv2d(self.num_in_ftrs, self.num_conv_ftrs, 3, padding = 1)
    self.bn_conv = nn.BatchNorm2d(self.num_conv_ftrs)

  def forward(self, input):

    # Gabor filtering of input
    gabor_feature = \
      self.bn_gabor(F.max_pool2d(self.gabor_feature_module(input), 
        self.pool_size, 
        stride=self.pool_size))
    
    # Apply Convolution filter to input
    # Remove ReLU for convolution layer before fully connected layer
    if self.is_last:
      conv_feature = self.bn_conv(F.max_pool2d(self.conv(input), 
        self.pool_size, 
        stride=self.pool_size))
    else:
      conv_feature = self.bn_conv(F.max_pool2d(F.relu(self.conv(input)), 
        self.pool_size, 
        stride=self.pool_size))
    
    # Construction of output by concatenation of features 
    # from Gabor filtering and convolution layer
    final_feature = torch.cat([conv_feature, gabor_feature], dim=1)

    return final_feature

class ProposedNetwork(Module):

  def __init__(self, type):
    """
    Proposed network for age estimation
    We used 6 mixed block: TGW layer + 3x3 Convolution layer
    Number of features from TGW decreases as layers going deeper
    Number of features from 3x3 convolution decreases as layers going deeper
    
    Args:
      type: operating mode of proposed network. 
            it can be set as one of following 4 choices
            
        'all'   : learns sampling grids and parameters of Gabor wavelets
        'ksize' : learns sampling grid (Best Method, abbreviation=Kernel SIZE)
        '~ksize': learns other parameters of Gabor wavelets except orientation
        'none'  : do not learn any parameter of Gabor wavelets but steering 
                  by 1x1 convolution is enabled.

    """
    super(ProposedNetwork, self).__init__()

    b1_preset = {'ksize':6, 'sig':5.4, 'gamma':0.3, 'lambd':6.8}
    self.b1 = GaborBlockModuleDagn(48, 2, 3, 256, 256, 
        b1_preset, type, False)
    b2_preset = {'ksize':5, 'sig':4.5, 'gamma':0.3, 'lambd':5.6}
    self.b2 = GaborBlockModuleDagn(24, 2, 512, 256, 256, 
        b2_preset, type, False)
    b3_preset = {'ksize':4, 'sig':3.6, 'gamma':0.3, 'lambd':4.6}
    self.b3 = GaborBlockModuleDagn(12, 2, 512, 256, 256, 
        b3_preset, type, False)
    b4_preset = {'ksize':3, 'sig':2.8, 'gamma':0.3, 'lambd':3.5}
    self.b4 = GaborBlockModuleDagn(6, 2, 512, 128, 128, 
        b4_preset, type, True)

    self.linA_1 = nn.Linear(2304, 512) #4095  2304
    self.linA_1_drop = nn.Dropout2d(p=0.5)
    self.linA_2 = nn.Linear(512, 512)
    self.linA_2_drop = nn.Dropout2d(p=0.5)
    self.linA_3 = nn.Linear(512, 51)
    self.linA_3_drop = nn.Dropout2d(p=0.5)
    
    
    self.linB_1 = nn.Linear(2304, 512) #4095  2304
    self.linB_1_drop = nn.Dropout2d(p=0.5)
    self.linB_2 = nn.Linear(512, 512)
    self.linB_2_drop = nn.Dropout2d(p=0.5)
    self.linB_3 = nn.Linear(512, 51)
    self.linB_3_drop = nn.Dropout2d(p=0.5)
    
    
    self.linC_1 = nn.Linear(2304, 512) #4095  2304
    self.linC_1_drop = nn.Dropout2d(p=0.5)
    self.linC_2 = nn.Linear(512, 512)
    self.linC_2_drop = nn.Dropout2d(p=0.5)    
    self.linC_3 = nn.Linear(512, 51)
    self.linC_3_drop = nn.Dropout2d(p=0.5)
    
    
    self.lin_4 = nn.Linear(153, 51)
    
    self.lrelu = nn.LeakyReLU(0.01);
    self.bn = nn.BatchNorm2d(153)

  def forward(self, in1, in2, in3):

    #1st branch  
    output1 = self.b1(in1)
    output1 = self.b2(output1)
    output1 = self.b3(output1)
    output1 = self.b4(output1)    

    output1 = output1.view(output1.size()[0],-1)
    output1 = self.linA_1_drop(F.relu( self.linA_1(output1) ))    
    output1 = self.linA_2_drop(F.relu(self.linA_2(output1)) )
    output1 = self.linA_3_drop(F.relu(self.linA_3(output1)) )
    
    #2st branch  
    output2 = self.b1(in2)
    output2 = self.b2(output2)
    output2 = self.b3(output2)
    output2 = self.b4(output2)    

    output2 = output2.view(output2.size()[0],-1)
    output2 = self.linB_1_drop(F.relu( self.linB_1(output2) ))
    output2 = (F.relu( self.linB_2(output2) ))
    output2 = self.linB_3_drop( F.relu(self.linB_3(output2)) )
    
    #3rd branch  
    output3 = self.b1(in3)
    output3 = self.b2(output3)
    output3 = self.b3(output3)
    output3 = self.b4(output3)    

    output3 = output3.view(output3.size()[0],-1)
    output3 = self.linC_1_drop(F.relu( self.linC_1(output3) ))
    output3 = (F.relu( self.linC_2(output3) ))
    output3 = self.linC_3_drop(F.relu(self.linC_3(output3)) )
    
    output = torch.cat((output1, output2, output3),1);
    output = self.lin_4(F.relu(output) );

    return output;
    
def init_weights(m):
  """ Xavier initialization for convolution layers and FC layers"""

  if type(m) == nn.Conv2d or type(m) == nn.Linear:
    nn.init.xavier_normal(m.weight.data)

if __name__ == '__main__':
  print('test')