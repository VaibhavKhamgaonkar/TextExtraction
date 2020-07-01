""" 
Content of PreProcessing 

"""

# -*- coding: utf-8 -*-
"""
Created on Mon May  4 10:02:25 2020

@author: khamgaov
"""

import numpy as np, pandas as pd, os, imutils, cv2, time
from tqdm import tqdm 
import glob
import re, sys
import pytesseract
from PIL import Image
from skimage import exposure
from skimage import feature
from imutils import perspective
from scipy.spatial import distance
from scipy import spatial
from collections import Counter
from imutils.contours import sort_contours





class Preprocessing():
    
    '''
    place for all the abstact functions
    Arguments:
        baseDirPath: base dir path
        word_list_path : path of technical text corpus whic is used for spell checking purpose
    
    '''
    def __init__(self, baseDirPath, word_list_path):
        self.path = baseDirPath
        #self.outputPath = path + 'Temp_Page_images/'
        
        self.ticksPath = baseDirPath + 'templates/ticks/'
        
        #word_list_path = baseDirPath + '/templates/
        
        
        tickTemplate = cv2.imread(self.ticksPath + '1.png')
        tickTemplate = cv2.resize(tickTemplate, (32,32), cv2.INTER_AREA)
        
        self.defaultTickFeatures, hogImage = feature.hog(tickTemplate, orientations=9, pixels_per_cell=(9, 9),
            		cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2", visualize=True)
        
        def words(text): return re.findall(r'\w+', text)
        self.WORDS = Counter(words(open(word_list_path,encoding='utf8').read()))

    #--------------------------------------------------------------------------
    
    @staticmethod 
    def order_points(pts):
        rect = np.zeros((4, 2), dtype = "float32")
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
    	# return the ordered coordinates
        return rect
    #---------------------------------------------------- 
    @staticmethod 
    def four_point_transform(image, pts):
        rect = Preprocessing.order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
    		[0, 0],
    		[maxWidth - 1, 0],
    		[maxWidth - 1, maxHeight - 1],
    		[0, maxHeight - 1]], dtype = "float32")
    	# compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    	# return the warped image
        return warped
    #----------------------------------------------------  
    @staticmethod
    def sliding_window(image, stepSize, windowSize = (256,256)):
        # slide a window across the image
     	for y in range(0, image.shape[0], stepSize):
             for x in range(0, image.shape[1], stepSize):
     			# yield the current window
                 yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
            
    #----------------------------------------------------
    
    def thresholdingWithSlidingWindow(self, img, stepSize=64, windowSize = (256,256),
                                      blkSize = 111, C=39, visualizeImg = False):
        
        ''' 
        Custom Threshholding function to substract the background fro mthe image using sliding window
        technique.
        Arguments:
            1. img : image file ( 3 channel image or RGB)
            2. stepSize : step Size for Sliding window
            3. windowSize : size of sliding window
            4. blkSize: Block size for adaptive threshold method for calculating the background
            5. C: consatnt value; subtracted from the mean or weighted mean in the surrounding region.
            6. visualizeImg : Whether to the operation in action.
        
        '''
        img = cv2.resize(img,  None, fx = 1, fy = 1, interpolation = cv2.INTER_CUBIC)
        imgBackup = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        winW, winH = windowSize
        for (x, y, window) in Preprocessing.sliding_window(img, stepSize= stepSize, windowSize=windowSize):
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            clone = img.copy()
            
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            roi = img[y: y+winH, x:x + winW ]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            #cropped = cv2.medianBlur(roi.copy(), 3)
            #cropped = cv2.GaussianBlur(original[y: y+winH, x:x + winW ], (3,3),0)
            #cropped = cv2.fastNlMeansDenoising(original[y: y+winH, x:x + winW ],None,10,7,21)
            roi = cv2.GaussianBlur(roi, (3,3),0)
            #_, roi = cv2.threshold(roi.copy(), 0,255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
            roi = cv2.adaptiveThreshold(roi.copy(), 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                            cv2.THRESH_BINARY_INV, blkSize, C)
            
            
            #img[y: y+winH, x:x + winW ] = cv2.cvtColor(roi,cv2.COLOR_GRAY2BGR)
            imgBackup[y: y+winH, x:x + winW ] = roi
            
            if visualizeImg:
                cv2.imshow("original_Img", clone)
                cv2.imshow("CroppedWindow", roi)
                #cv2.imshow("cropped", roi)
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
                #time.sleep(0.025)
            
        cv2.destroyAllWindows()
            
        return imgBackup
    
    #----------------------------------------------------
    
    def cleanImage(self, image):
        ''' 
        clean the image with random ticks and other Noise.
        
        '''
        img = image.copy()
        lower_black = np.array([0,0,0], dtype = "uint16")
        upper_black = np.array([100,100,100], dtype = "uint16")
        black_mask = cv2.inRange(img, lower_black, upper_black)
        
        kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (7,5), )
        black_mask = cv2.dilate(black_mask, kernel=kernal, iterations=2 )
        cv2.imshow('black_mask',black_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        testImg = cv2.cvtColor(np.ones(shape=img.shape, dtype = 'uint8')*255, cv2.COLOR_BGR2GRAY)
        #print(testImg.shape)
        
        #black_mask = cv2.bitwise_not(black_mask)
        # cv2.imshow('white_mask',black_mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # img = cv2.bitwise_and(img, img, mask = black_mask)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.GaussianBlur(img, (3,3),0)
        # _,thres = cv2.threshold(img.copy(), 0,255, 
        #                cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 
        # cv2.imshow('white_mask',img)
        # cv2.imshow('testImg',thres)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        ''' get the contours from the image '''
        cnts = cv2.findContours(black_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        for cnt in cnts:
            #a = black_mask.copy()
            area = cv2.contourArea(cnt)
            if area < 200:
                continue
            temp = np.zeros(shape=image.shape[:2], dtype = 'uint8')
            x,y,w,h = cv2.boundingRect(cnt)
            
            cv2.rectangle(temp, (x,y), (x+w,y+h),(255),-1,)
            
            roi = img[y:y+h, x:x+w]#cv2.bitwise_and(img,img, mask=temp)
            
            testImg[y:y+h, x:x+w] = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            
            testImg[testImg>125] = 255
            cv2.imshow('a',roi)
            cv2.imshow('final1',testImg)
            cv2.waitKey(35)
        cv2.destroyAllWindows()
            
        #     rect = cv2.minAreaRect(contour)
        #     box = cv2.boxPoints(rect)
        
        #     ext_left = tuple(contour[contour[:, :, 0].argmin()][0])
        #     ext_right = tuple(contour[contour[:, :, 0].argmax()][0])
        #     ext_top = tuple(contour[contour[:, :, 1].argmin()][0])
        #     ext_bot = tuple(contour[contour[:, :, 1].argmax()][0])
        
        #     roi_corners = np.array([box], dtype=np.int32)
        
        #     cv2.polylines(img, roi_corners, 1, (255, 0, 0), 3)
        #     cropped_image = image[ext_top[1]:ext_bot[1], ext_left[0]:ext_right[0]]
            
            
        #     #
        #     # ''' crop from the original image and add in in testImage '''
            
        #     # crop = image[y:y+h, x:x+w]
        #     # print(crop.shape)
        #     #testImg[ext_top[1]:ext_bot[1], ext_left[0]:ext_right[0]] = cropped_image 
        #     #cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            
        #     cv2.imshow('crop',cropped_image)
        #     cv2.imshow('testImg',testImg)
        #     cv2.waitKey(0)
        # cv2.destroyAllWindows()
            
            
        
        
        
        ''' superimposing the img on test img '''
        # img = cv2.addWeighted(testImg,0.5,img, 0.5,0)
        # img[img>200]=255
        #img = cv2.bitwise_xor(img,image, mask = img)
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,5))
        #lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        #outliers = np.where((img[:,:,0]>160) | (img[:,:,1]>180) | (img[:,:,2]>180))
        #img[outliers] = (255,255,255)
        #img[:,:,0] = np.zeros([img.shape[0], img.shape[1]])
        #img[:,:,2] = np.zeros([img.shape[0], img.shape[1]])
        #img = cv2.bitwise_not(img)
        #img = cv2.cvtColor(hsv, cv2.COLOR_LAB2BGR)
        #lab[lab<150]=0
        
        """ Can be used if the image has hand written noise  such as marking or tick marks"""
       #  g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       #  #g = cv2.bilateralFilter(g.copy(), 9, 75, 75)
       #  g = cv2.GaussianBlur(g.copy(), (3,3), 0)
       #  #gX = cv2.Sobel(g, ddepth=cv2.CV_64F, dx=1, dy=0)
       #  #gY = cv2.Sobel(g, ddepth=cv2.CV_64F, dx=0, dy=1)
        
       #  #gX = cv2.convertScaleAbs(gX)
       #  #gY = cv2.convertScaleAbs(gY)
         
       #  # combine the sobel X and Y representations into a single image
       # # sobelCombined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
       #  _,thres = cv2.threshold(g.copy(), 0,255, 
       #                cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 
        
       #  cnts = cv2.findContours(thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       #  cnts = imutils.grab_contours(cnts)
        
       #  test = img.copy()
       #  cv2.imshow('Clean',test)
       #  cv2.waitKey(0)
       #  cv2.destroyAllWindows()
        
       #  for c in cnts:
       #      box = cv2.minAreaRect(c)
       #      box = cv2.boxPoints(box)
       #      box = np.array(box,dtype="int")
       #      box = perspective.order_points(box)
       #      (tl, tr, br, bl) = box.astype("int")
       #      h = distance.euclidean(tl,bl)
       #      w = distance.euclidean(tl,tr)
       #      ar = np.round(h/w, 3)
       #      if  ar <= 0.5:
       #          #get the features from ROI
       #          wrapped = Preprocessing.four_point_transform(test, box.astype("float"))
       #          wrapped = cv2.resize(wrapped, (32,32),cv2.INTER_AREA)
       #          yFeat, hogImage = feature.hog(wrapped, orientations=9, pixels_per_cell=(9, 9),
       #      		cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2", visualize=True)
       #          #checkign the features with yTemplate
       #          similarity = 1 - spatial.distance.cosine(self.defaultTickFeatures, yFeat)
       #          if similarity > 0.5:
       #              cv2.drawContours(test, [box.astype("int")], -1, (255,255,255), cv2.FILLED)
       #              #cv2.rectangle(test,(x,y),(x+w,y+h),(255,255,255),-1)
       #              # cv2.putText(test, str(np.round(similarity, 3)), tuple(box[0].astype("int")), cv2.FONT_HERSHEY_COMPLEX, 
       #              #           0.4, (0,0,255))
        
       #  test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
       #  test = cv2.bitwise_not(test)
       #  test[test<150] = 0
       #  test[test>10] = 255
        
       #  kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (10,6), )
       #  test = cv2.dilate(test, kernel=kernal, iterations=2 )
        
       #  #_,test = cv2.threshold(test.copy(), 0,255, cv2.THRESH_BINARY)
       #  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
       #  final = cv2.bitwise_and(image, image, mask = test)
       #  cv2.imshow('mask',test)
        cv2.imshow('final',testImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()            
        #return cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(testImg, cv2.COLOR_GRAY2BGR)
    #--------------------------------------------------------------------------
    
    def spellCheck(self, word):
        """def words(text): return re.findall(r'\w+', text.lower())"""

        """WORDS = Counter(words(open(self.wordsFilePath).read()))"""
        
        # if '\\n' in word:
        #     flag = '\\n'
        # else:
        #     flag = ''
        def P(word, N=sum(self.WORDS.values())): 
            "Probability of `word`."
            return self.WORDS[word] / N
        
        def correction(word): 
            "Most probable spelling correction for word."
            return max(candidates(word), key=P)
        
        def candidates(word): 
            "Generate possible spelling corrections for word."
            return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])
        
        def known(words): 
            "The subset of `words` that appear in the dictionary of WORDS."
            return set(w for w in words if w in self.WORDS)
        
        def edits1(word):
            "All edits that are one edit away from `word`."
            letters    = 'abcdefghijklmnopqrstuvwxyz'
            splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
            deletes    = [L + R[1:]               for L, R in splits if R]
            transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
            replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
            inserts    = [L + c + R               for L, R in splits for c in letters]
            return set(deletes + transposes + replaces + inserts)
        
        def edits2(word): 
            "All edits that are two edits away from `word`."
            return (e2 for e1 in edits1(word) for e2 in edits1(e1))
       
        return correction(word)


#==============================================================================
         

if __name__ == '__main__':
    
    path = 'D:\\Temp\\'
    pytesseract.pytesseract.tesseract_cmd = path + 'tesseract/tesseract.exe'
    obj = Preprocessing(baseDirPath = path, word_list_path=path+'templates/finalCorpus.txt')
    
    st = time.time()
    obj.processPDF(uploadPath=path+'static/uploads/')
    et = time.time()
    print(f'Finished processing in {np.round((et-st)/60.0, 3)} mins.')
    
    
   