# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 12:35:17 2020

@author: VaibhavK
@description: Extraction of Text 
"""


import cv2, numpy as np, pandas as pd, os, time, imutils
import pytesseract
import logging
from multiprocessing import Pool
import re
from Helper_Scripts.Preprocessing import Preprocessing
from tqdm import tqdm


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
fileHandler = logging.FileHandler('./Logs/logs.log')
fileHandler.setFormatter(formatter)
    
logger.addHandler(fileHandler)


class ExtractData(Preprocessing):
    """ 
    Extract the text out of Scanned images
    
    Arguements:
        1. baseDirPath = path of base dir
        2. WordlistPath = text corpus path for spell checking 
        3. pytesseractPath = path of pytesseract exe file
    """
    def __init__(self, baseDirPath, wordListPath, 
                     pytesseractPath):
        
        pytesseract.pytesseract.tesseract_cmd = pytesseractPath
        #super(Preprocessing, self).__init__()
        super().__init__(baseDirPath = baseDirPath, word_list_path=wordListPath)
        
    #-----------------------------------------------------------------------
    
    def spellChecking(self, text):
        """ For spell checking 
        
        Argument:
            text : Text on which spell checking needs to perform.    
            
        """
        temp = []
        for item in text.split('\n'):
            temp.append(' '.join([self.spellCheck(word) if (word.isalpha() and len(word) > 3)  else word for word in item.split()]))
            #temp.append(' '.join([self.spellCheck(word) if len(word) > 3  else word for word in item.split()]))
            
        return '\n'.join(temp)
    #--------------------------------------------------------------------------

    def readImgs(self, imgPath, visualizeImg=False, isSpellCheckRequired=False, stepSize = 64, 
                                                  windowSize=(256,256), blkSize=111, C=39,
                                                  psm=6, oem=0):
        ''' 
        Extract data from the scanned pdf
        argument:
            1. imgPath = path of img file to be extracted
            2. visualizeImg = Bool; whether to visualize the img after 
                performing clean operation operation 
            3. isSpellCheckRequired = whether to perform spell checking operations
                                        for extracted text
            4. stepSize: What pixel steps the winodw sould move; mainly used in slinding window technique 
            5. windowSize: Size of sliding window: 
                
            Sliding window technique is mainly used to get the proper Threhold (binary) image with 
            background removed. 
            6. blkSize: Block size for adaptive threshold method for calculating the background
            7. C: consatnt value; subtracted from the mean or weighted mean in the surrounding region.
            8. psm : page segmentation mode for OCR operation default is 6
            9. oem: pretrained model code used for extracting the text from image
                    0    Legacy engine only.
                    1    Neural nets LSTM engine only.
                    2    Legacy + LSTM engines.
                    3    Default, based on what is available.
                    
                    https://nanonets.com/blog/ocr-with-tesseract/
        '''
        # print(f'Starting ====== {imgPath}')
        logger.info(f'Starting ====== {imgPath}')
        page = cv2.imread(imgPath)
        text = ' '
        returnFlag = 'WithOutSpellCheck'
        ''' Clean the page : Needed when there are any hand writtent text or tickmarks on the Image '''
        
        logger.info('performing cleaning operation..')
        page = self.cleanImage(page)
        logger.info(f'Image is cleaned and has shape {page.shape}')
        
        # cv2.imshow('Cleaned', page)
        # # cv2.imshow('ResizeImg', imutils.resize(page,width=620))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #page = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        #_, page = cv2.threshold(page.copy(), 0,255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        #page = cv2.adaptiveThreshold(page.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        #                             cv2.THRESH_BINARY_INV, 85,21)
       
        page = self.thresholdingWithSlidingWindow(img= page,stepSize = stepSize, 
                                                  windowSize=windowSize,
                                                  visualizeImg = visualizeImg,
                                                  blkSize = blkSize,
                                                  C = C)
        if visualizeImg:
            #cv2.imshow('Img', page)
            cv2.imshow('ResizeImg', imutils.resize(page,width=620))
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
        logger.info('Now Extracting text')
        text += ' ' + pytesseract.image_to_string(page,
                                                      config = f'--psm {psm}, --oem {oem}' )
                          #-c preserve_interword_spaces=1x1,
        
        ''' Perform spellcheck operation '''
        if isSpellCheckRequired:
            logger.info('applying Spell checking')
            text = self.spellChecking(text)
            returnFlag = 'WithSpellCheck'
            
        return text,returnFlag 



#=============================================================================


if __name__ == '__main__':
    
    pytesseractPath = 'D:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    
    obj = ExtractData(baseDirPath ='D:\\!@Projects\\FinicityTask\\',
                      wordListPath = 'D:\\!@Projects\\FinicityTask\\templates\\big.txt',
                      pytesseractPath = pytesseractPath
                      )    



    text, returnFlag = obj.readImgs('D:\\!@Projects\\FinicityTask\\Data\\Img3.png',
                                    visualizeImg=True,
                                   isSpellCheckRequired=True)
    print(text)


