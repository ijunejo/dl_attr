# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 11:04:38 2018

@author: imran
"""
import cv2 
import numpy as np
from os import listdir
from os.path import isfile, join, isdir

import pickle
import random as rd
from scipy.io import loadmat

class RAP_DATA_RGB_144x48:
    
    def __init__(self, args):
        self.path = args['path']
        self.width = args['width']
        self.height = args['height']
        self.bpp = args['bpp']
        self.attrs = args['attrs']
        self.pckl_filename = args['pckl_name'];
        
        self.filenames = [];
              
        #select file name based on the no. of attributes
        self.pckl_filename += '_51attrs.pckl'                
                
        if isdir(self.path) or isfile(self.pckl_filename):   
            data = loadmat(open('RAP_annotation.mat', 'rb'))        
            self.labels = np.array(data['RAP_annotation'][0][0]['label'], dtype='bool')            
                    
            for idx in range( self.labels.shape[0] ):
                self.filenames.append( data['RAP_annotation'][0][0]['imagesname'][idx][0][0] );          
            
            self.data = np.zeros((self.labels.shape[0], self.height, self.width, self.bpp) , dtype ='uint8')
            
            if not isfile(self.pckl_filename): #if the data not already processed
                self.prepare_data();
                print('creating file ...' , self.pckl_filename)
                f_data = open(self.pckl_filename, 'wb')
                pickle.dump([self.labels, self.data], f_data)
                f_data.close()
            else: #load the pickle file
                print('loading file ...' , self.pckl_filename)
                f_data = open(self.pckl_filename, 'rb')
                self.labels, self.data = pickle.load(f_data);
                f_data.close();
                        
        else:
            print("__init__ error: folder not found")
            raise Exception()          
        
    def prepare_data(self):         
         self.idx_ctr = 0;
         
         #for each dir, go and read the images there using the labels.txt file
         for file_idx in self.filenames:
             fill_filename = join(self.path,file_idx)
             #print('reading file: ' , self.idx_ctr)
             #read file using opencv
             Img = cv2.imread( fill_filename )
             Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB);
                                        
             self.data[self.idx_ctr,:,:,:] = Img;
             self.idx_ctr += 1;    
        
    def load_data(self):
        # we want to return train, validation, and test data with:
        #9500, 1900, 7600        
        tot_dSize = list(range( self.data.shape[0] ));
        
        #shuffle indices twice (just in case)
        rd.shuffle( tot_dSize );
        
        self.data = self.data[tot_dSize]
        self.labels = self.labels[tot_dSize];
        self.labels = self.labels[:,0:self.attrs];
        
        return (self.data[0:33268,:],self.labels[0:33268,:]),(self.data[33268:,:],self.labels[33268:,:])

#create the info for reading the dataset     
if __name__ == "__main__":
        
    args_data = {};
    args_data['path'] = '\\RAP_dataset'
    args_data['width'] = 48
    args_data['height'] = 144
    args_data['bpp'] = 3
    args_data['pckl_name'] = 'rap_data_rgb_144x48'
    args_data['attrs'] = 51
    #
    ##load data
    RAP40K = RAP_DATA_RGB_144x48(args_data)
    (x_train, y_train), (x_test, y_test) = RAP40K.load_data( );