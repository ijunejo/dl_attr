# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 11:04:38 2018

@author: imran
"""
import cv2 
import numpy as np
from os import listdir
from os.path import isfile, join, isdir
import fnmatch
import pickle
import random as rd

class Peta_CUHK_144x48:
    
    def __init__(self, args):
        self.path = args['path']
        self.width = args['width']
        self.height = args['height']
        self.bpp = args['bpp']
        self.attrs = args['attrs']
        self.pckl_filename = args['pckl_name'];
        
        #select file name based on the no. of attributes
        if self.attrs != 35:
            self.pckl_filename += '_all.pckl'                
        else:
            self.pckl_filename += '_35attrs.pckl'                
                
        if isdir(self.path) or isfile(self.pckl_filename):
            if self.attrs != 35:                
                temp_labels = 'accessoryHeadphone personalLess15 personalLess30 personalLess45 personalLess60 personalLarger60 carryingBabyBuggy carryingBackpack hairBald footwearBoots lowerBodyCapri carryingOther carryingShoppingTro carryingUmbrella lowerBodyCasual upperBodyCasual personalFemale carryingFolder lowerBodyFormal upperBodyFormal accessoryHairBand accessoryHat lowerBodyHotPants upperBodyJacket lowerBodyJeans accessoryKerchief footwearLeatherShoes upperBodyLogo hairLong lowerBodyLongSkirt upperBodyLongSleeve lowerBodyPlaid lowerBodyThinStripes carryingLuggageCase personalMale carryingMessengerBag accessoryMuffler accessoryNothing carryingNothing upperBodyNoSleeve upperBodyPlaid carryingPlasticBags footwearSandals footwearShoes hairShort lowerBodyShorts upperBodyShortSleeve lowerBodyShortSkirt footwearSneakers footwearStocking upperBodyThinStripes upperBodySuit carryingSuitcase lowerBodySuits accessorySunglasses upperBodySweater upperBodyThickStripes lowerBodyTrousers upperBodyTshirt upperBodyOther upperBodyVNeck footwearBlack footwearBlue footwearBrown footwearGreen footwearGrey footwearOrange footwearPink footwearPurple footwearRed footwearWhite footwearYellow hairBlack hairBlue hairBrown hairGreen hairGrey hairOrange hairPink hairPurple hairRed hairWhite hairYellow lowerBodyBlack lowerBodyBlue lowerBodyBrown lowerBodyGreen lowerBodyGrey lowerBodyOrange lowerBodyPink lowerBodyPurple lowerBodyRed lowerBodyWhite lowerBodyYellow upperBodyBlack upperBodyBlue upperBodyBrown upperBodyGreen upperBodyGrey upperBodyOrange upperBodyPink upperBodyPurple upperBodyRed upperBodyWhite upperBodyYellow';
            else:                
                temp_labels = 'personalLess30 personalLess45 personalLess60 personalLarger60 carryingBackpack carryingOther lowerBodyCasual  upperBodyCasual lowerBodyFormal upperBodyFormal accessoryHat upperBodyJacket lowerBodyJeans footwearLeatherShoes upperBodyLogo hairLong personalMale carryingMessengerBag accessoryMuffler accessoryNothing carryingNothing upperBodyPlaid carryingPlasticBags footwearSandals footwearShoes lowerBodyShorts upperBodyShortSleeve lowerBodyShortSkirt footwearSneakers upperBodyThinStripes accessorySunglasses lowerBodyTrousers upperBodyTshirt upperBodyOther upperBodyVNeck';
            
            self.attrs = temp_labels.split();
            self.reject_labels = ['accessoryFaceMask','lowerBodyLogo','accessoryShawl','lowerBodyThickStripes']
            self.tot_attrs = len(self.attrs);            
            self.labels = np.zeros( (19000, self.tot_attrs), dtype = 'uint8' )            
            self.data = np.zeros((19000, self.height, self.width, self.bpp) , dtype ='uint8')
            
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
         #get all the subfolder first
         sub_dirs = [f for f in listdir(self.path)]
         idx_ctr = 0;
         sums = 0;
         #for each dir, go and read the images there using the labels.txt file
         for each_dir in sub_dirs:
             curr_path = join(self.path,each_dir,'archive')
             
             files_in_sub = [f for f in listdir(curr_path) if isfile(join(curr_path,f))]
             ctr = 0;
             #open the Label.txt file in folder;
             with open( join(curr_path,'Label.txt'), 'r' ) as fileObj:
                 for one_line in fileObj: #each line
                     line_tokens = one_line.strip().split(); #break the labels
                     if each_dir == 'CUHK':
                         pattern = line_tokens[0];
                     else:  
                         pattern = line_tokens[0] + '_*.*';
                     
                     label_row = np.zeros( self.tot_attrs, dtype = 'int16' );                 
                     #for each label, find the index in the self.attrs and make one-hot
                     for attr_name in line_tokens[1:]:
                         #if attr_name not in self.reject_labels:
                         if attr_name in self.attrs:
                             tmp_index = self.attrs.index(attr_name);
                             label_row[tmp_index] = 1;
                     
                     #now find images per the 1st value in each line
                     imgs_per_ID = fnmatch.filter(files_in_sub, pattern);
                     #add this images to the training along with the labels
                     for each_file in imgs_per_ID:
                         ctr += 1;
                         #read file using opencv
                         Img = cv2.imread( join(curr_path, each_file) )
                         Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB);                                           
                         
                         self.labels[idx_ctr,:] = label_row;                         
                         self.data[idx_ctr,:,:,:] = Img;
                         idx_ctr += 1;
             print('read files in the folder' + curr_path + ': ' + str(ctr) )
             sums += ctr;             
         
         print('total size ' + str(idx_ctr))
         print('total sums ' + str(sums))
         return sub_dirs
     
    def load_data(self, validation_separate):
        # we want to return train, validation, and test data with:
        #9500, 1900, 7600        
        tot_dSize = list(range( self.data.shape[0] ));
        
        #shuffle indices twice (just in case)
        rd.shuffle( tot_dSize );
        rd.shuffle( tot_dSize );
        rd.shuffle( tot_dSize );
        
        self.data = self.data[tot_dSize]
        self.labels = self.labels[tot_dSize];
        
        if validation_separate == True:            
            return (self.data[0:9500,:],self.labels[0:9500,:]),(self.data[11400:,:],self.labels[11400:,:]),(self.data[9500:11400,:],self.labels[9500:11400,:])
        else:
            return (self.data[0:9500,:],self.labels[0:9500,:]),(self.data[11400:,:],self.labels[11400:,:])


#create the info for reading the dataset     
if __name__ == "__main__":
        
    args_data = {};
    args_data['path'] = '\\PETA dataset'
    args_data['width'] = 48
    args_data['height'] = 144
    args_data['bpp'] = 3
    args_data['pckl_name'] = 'peta_data_144x48'
    args_data['attrs'] = 35
    #
    ##load data
    PETA19k = Peta_CUHK_144x48(args_data)
    (x_train, y_train), (x_test, y_test), (x_val, y_val)= PETA19k.load_data(validation_separate = True);



