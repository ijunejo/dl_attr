# -*- coding: utf-8 -*-
from __future__ import print_function, division

#import cv2
import numpy as np
import warnings
from PIL import Image
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

# Ignore warnings
warnings.filterwarnings("ignore")

class PETADataset(Dataset):
  
  def __init__(self, pkl_filename, transforms=None, reqtype='train'):
    """
    Dataset for Adience Database

    Args:
      image_dir (List of string): Directory with all the images.
      transforms: transforms could be on the input images
    """
    self.width = 48
    self.height = 144
    self.bpp = 3
    self.attrs = 35
    self.pckl_filename = pkl_filename
            
    self.image_files = []
    self.targets = []
    self.transforms = None
    
    #load the pickel file: created by the: read_RAP_own_model_144x48_rgb
    print('loading file ...' , self.pckl_filename)
    f_data = open(self.pckl_filename, 'rb')
    self.targets, self.image_files = pickle.load(f_data);
    self.targets = self.targets[:,0:self.attrs];
    f_data.close();
    
    if reqtype=='train':
        self.image_files = self.image_files[0:9500,:];
        self.targets = self.targets[0:9500,:];
        
    if reqtype == 'test':
        self.image_files = self.image_files[11400:,:];
        self.targets = self.targets[11400:,:];
        
    if (transforms != None):
      self.transforms = transforms

  def get_class_wt(self):
      per_class_ct = self.targets.sum(0)
      neg_score = self.targets.shape[0] - per_class_ct
      
      return neg_score/(per_class_ct + 1e-5);
      
  def __len__(self):
    return len(self.image_files)

  def __getitem__(self, idx):
    #orig 144x48_3channel
    img = self.image_files[idx]

    targets = torch.tensor(self.targets[idx], dtype=torch.float32)
    
    #create three imgae inputs
    input1 = Image.fromarray(img[:48,:,:]).convert('RGB');
    input2 = Image.fromarray(img[48:96,:,:]).convert('RGB');
    input3 = Image.fromarray(img[96:144,:,:]).convert('RGB');
    
    if (self.transforms != None):
      input1 = self.transforms(input1)
      input2 = self.transforms(input2)
      input3 = self.transforms(input3)
    
    sample = (input1, input2, input3, targets)

    return sample