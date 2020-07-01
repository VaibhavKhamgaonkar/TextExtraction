# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:13:35 2020

@author: asus
"""


import os, time, configparser
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import logging
from Helper_Scripts.ExtractData import ExtractData



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
fileHandler = logging.FileHandler('./Logs/logs.log')
fileHandler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
logger.addHandler(fileHandler)
logger.addHandler(stream_handler)


class Main(ExtractData):
    """
        1. baseDirPath = path of base dir
        2. WordlistPath = text corpus path for spell checking 
        3. pytesseractPath = path of pytesseract exe file
        4. conf: object of config parser require to access the config file.
    """
    
    def __init__(self, baseDirPath, wordListPath, pytesseractPath, conf):
        
        self.conf = conf
        
        super().__init__(baseDirPath=baseDirPath, 
                      wordListPath=wordListPath, 
                      pytesseractPath=pytesseractPath)
        
    #-------------------------------------------------------------------------
    
    def mainFunction(self, imgPath):
        ''' 
        Extract the Data from images and save it in text file
        Arguments: 
            1. imgPath: path of image to be extracted
                   
        '''
        visiualize = eval(self.conf.get('visiualize', 'visiualize'))
        windowSize = eval(self.conf.get('images', 'windowSize'))
        stepSize = eval(self.conf.get('images', 'stepSize'))
        isSpellCheckRequired = eval(self.conf.get('operations', 'isSpellCheckRequired'))
        blkSize = eval(self.conf.get('images', 'adaptiveThreshold_Blk_size'))
        C = eval(self.conf.get('images', 'adaptiveThreshold_C'))
        psm = eval(self.conf.get('OCRConfig', 'psm'))
        oem = eval(self.conf.get('OCRConfig', 'oem'))
            
        text, returnFlag = self.readImgs(imgPath=imgPath, visualizeImg=visiualize, 
                                        isSpellCheckRequired=isSpellCheckRequired, 
                                        stepSize = stepSize, 
                                        windowSize=windowSize,
                                        blkSize=blkSize,
                                        C=C,
                                        psm=psm,
                                        oem=oem
                                        )
         
        if returnFlag == 'WithSpellCheck':
            name= 'WithSpellCheck_'
        else:
            name= 'WithOutSpellCheck_'
                                       
        ''' Write the output exctrated text in Output directory '''
        with open(f"{self.conf.get('paths', 'outputDir')}{name}_{imgPath.split('/')[-1]}.txt", 'w', encoding='utf-8') as f:
            f.write(text)
        
        #print(f"extraction process completed and file i stored at {self.conf.get('paths', 'outputDir')}{name}_{imgPath.split('/')[-1]}.txt")
        logger.info(f"\nExtraction process completed and file is stored at {self.conf.get('paths', 'outputDir')}{name}_{imgPath.split('/')[-1]}.txt")

#==============================================================================

if __name__ == '__main__':
    
    ''' Initialising Config file and reading the details '''
    parser = configparser.ConfigParser()
    parser.read('./config.ini')
    baseDirPath = parser.get('paths','baseDirPath')
    wordListPath = parser.get('paths','wordListPath')
    pytesseractPath = parser.get('paths','pytesseractPath')
    
    
    obj = Main(baseDirPath=baseDirPath, 
                      wordListPath=wordListPath, 
                      pytesseractPath=pytesseractPath,
                      conf = parser)
    
    
    """ Get all the images presen in the input data folder """
    
    files = [(baseDirPath + 'Data/' + item) for item in os.listdir(baseDirPath + 'Data/') if
             [i for i in ['.png', '.jpg','.jpeg'] if i in item]]
    
    
    
    for k, img in tqdm( enumerate(files), unit='imges'):
        # if k == 0:
        logger.info(obj.mainFunction(imgPath=img))
        #break
    

    """ For future developement : implimenting threading to boost the performance and reduce the 
    execution time using Multithreading and utilising cores of CPU.
    """
    # pool = Pool(processes=2)
    
    # pool.map(obj.mainFunction, files)
    #pool.close() 
    #pool.join()
    


    
    
    