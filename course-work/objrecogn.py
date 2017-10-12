# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np
import utilscv

# A Python class has been created, called ImageFeature
# which will contain for each of the images of the database,
# the information needed to compute object recognition.
class ImageFeature(object):
    def __init__(self, nameFile, shape, imageBinary, kp, desc):
        self.nameFile = nameFile
        self.shape = shape
        self.imageBinary = imageBinary
        self.kp = kp                    #KeyPoints
        self.desc = desc                #Descriptors
        #Matchings of the image of the database with the image of the webcam
        self.matchingWebcam = []
        #Matching the webcam with the current image of the database
        self.matchingDatabase = []
    #Allows to empty previously calculated matching for a new image
    def clearMatchingMutuos(self):
        self.matchingWebcam = []
        self.matchingDatabase = []

# Function for calculating, for each of the methods of calculation of features, 
# the features of each of the images of the directory "models"
def loadModelsFromDirectory():
    # The method returns a dictionary. The key is the features algorithm
    # while the value is a list of objects of type ImageFeature
    # where all the data of the features of the images of the

    dataBase = dict([
        ('SIFT', []), 
        ('AKAZE', []), 
        ('SURF', []), 
        ('ORB', []), 
        ('BRISK', [])]
    )
    #The number of features has been limited to 250 for the algorithm to flow.
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=250)
    akaze = cv2.AKAZE_create()
    surf = cv2.xfeatures2d.SURF_create(800)
    orb = cv2.ORB_create(400)
    brisk = cv2.BRISK_create()
    
    for imageFile in os.listdir("template"):
        colorImage = cv2.imread("template/" + str(imageFile))
        currentImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY)
        #We perform a resize of the image, so that the compared image is equal
        kp, desc = sift.detectAndCompute(currentImage, None)
        #Features are loaded
        dataBase["SIFT"].append(ImageFeature(imageFile, currentImage.shape, colorImage, kp, desc))
        
        kp, desc = akaze.detectAndCompute(currentImage, None)
        dataBase["AKAZE"].append(ImageFeature(imageFile, currentImage.shape, colorImage, kp, desc))
        
        kp, desc = surf.detectAndCompute(currentImage, None)
        dataBase["SURF"].append(ImageFeature(imageFile, currentImage.shape, colorImage, kp, desc))
        
        kp, desc = orb.detectAndCompute(currentImage, None)
        dataBase["ORB"].append(ImageFeature(imageFile, currentImage.shape, colorImage, kp, desc))
        
        kp, desc = brisk.detectAndCompute(currentImage, None)
        dataBase["BRISK"].append(ImageFeature(imageFile, currentImage.shape, colorImage, kp, desc))
    
    return dataBase
    

# Function responsible for calculating mutual matching, but nesting loops
# It is a very slow solution because it does not take advantage of Numpy power
# We do not even put a slider to use this method as it is very slow

def findMatchingMutuos(selectedDataBase, desc, kp):
    for imgFeatures in selectedDataBase:
        imgFeatures.clearMatchingMutuos()
        for i in range(len(desc)):
            primerMatching = None
            canditatoDataBase = None
            matchingSegundo = None
            candidateWebCam = None
            for j in range(len(imgFeatures.desc)):
                valorMatching = np.linalg.norm(desc[i] - imgFeatures.desc[j])
                if (primerMatching is None or valorMatching < primerMatching):
                    primerMatching = valorMatching
                    canditatoDataBase = j
            for k in range(len(desc)):
                valorMatching = np.linalg.norm(imgFeatures.desc[canditatoDataBase] - desc[k])
                if (matchingSegundo is None or valorMatching < matchingSegundo):
                    matchingSegundo = valorMatching
                    candidateWebCam = k
            if not candidateWebCam is None and i == candidateWebCam:
                imgFeatures.matchingWebcam.append(kp[i].pt)
                imgFeatures.matchingDatabase.append(imgFeatures.kp[canditatoDataBase].pt)
    return selectedDataBase

# Function responsible for calculating the mutual matching of a webcam image,
# with all the images of the database. Receive as input parameter
# the database based on the method of calculation of features used
# in the image input of the webcam.
def findMatchingMutuosOptimizado(selectedDataBase, desc, kp):
    for img in selectedDataBase:
        img.clearMatchingMutuos()
        for i in range(len(desc)):
            # The standard of difference of the current descriptor is calculated, with all
            # the image descriptors of the database. We got
            # without loops and making use of Numpy broadcasting, all distances
            # between the current descriptor with all the descriptors of the current image
            distanceListFromWebCam = np.linalg.norm(desc[i] - img.desc, axis=-1)
            #Obtain the candidate who is the shortest distance from the current descriptor
            candidatoDataBase = distanceListFromWebCam.argmin() 
            # It is checked if the matching is mutual, that is, if it is in the other direction. 
            # That is, it is verified that the has the current descriptor as best matching
            distanceListFromDataBase = np.linalg.norm(
                img.desc[candidatoDataBase] - desc,
                axis=-1
            )
             
            candidatoWebCam = distanceListFromDataBase.argmin()
            #If mutual matching is fulfilled, it is stored for later processing
            if (i == candidatoWebCam):
                img.matchingWebcam.append(kp[i].pt)
                img.matchingDatabase.append(img.kp[candidatoDataBase].pt)
        #For convenience they become Numpy ND-Array
        img.matchingWebcam = np.array(img.matchingWebcam)
        img.matchingDatabase = np.array(img.matchingDatabase)
    return selectedDataBase

# This function calculates the best image based on the number of inliers
# which has each image of the database with the image obtained from
# the web camera.
def calculateBestImageByNumInliers(selectedDataBase, projer, minInliers):
    if minInliers < 15:
        minInliers = 15
    
    bestIndex = None
    bestMask = None
    numInliers = 0
    
    for index, imgWithMatching in enumerate(selectedDataBase):
        #The RANSAC algorithm is computed to calculate the number of inliers
        _, mask = cv2.findHomography(
        	imgWithMatching.matchingDatabase,
        	imgWithMatching.matchingWebcam, 
        	cv2.RANSAC, 
        	projer
        )
        
        if not mask is None:
            # The number of inliers is checked from the mask.
            # If the number of inliers exceeds the minimum number of
            # and is a maximum (it has more inliers than the image
            # then it is considered that it is the image that matches the
            # stored in the database.
            countNonZero = np.count_nonzero(mask)
            if (countNonZero >= minInliers and countNonZero > numInliers):
                numInliers = countNonZero
                bestIndex = index
                bestMask = (mask >= 1).reshape(-1)
    # If an image has been obtained as the best image and, for
    # must have a minimum number of inlers, then calculate
    # the keypoints that are inliers from the mask obtained in
    # and is returned as the best image.
    if not bestIndex is None:
        bestImage = selectedDataBase[bestIndex]
        inliersWebCam = bestImage.matchingWebcam[bestMask]
        inliersDataBase = bestImage.matchingDatabase[bestMask]
        return bestImage, inliersWebCam, inliersDataBase
    return None, None, None
                
# This function calculates the affinity matrix A, paints a rectangle around
# of the detected object and paints in a new window the image of the database
# corresponding to the recognized object.
def calculateAffinityMatrixAndDraw(bestImage, inliersDataBase, inliersWebCam, imgout):
    #The affinity matrix A
    A = cv2.estimateRigidTransform(inliersDataBase, inliersWebCam, fullAffine=True)
    A = np.vstack((A, [0, 0, 1]))
    
    #Calculate the points of the rectangle occupied by the recognized object
    a = np.array([0, 0, 1], np.float)
    b = np.array([bestImage.shape[1], 0, 1], np.float)
    c = np.array([bestImage.shape[1], bestImage.shape[0], 1], np.float)
    d = np.array([0, bestImage.shape[0], 1], np.float)
    centro = np.array([float(bestImage.shape[0])/2, 
       float(bestImage.shape[1])/2, 1], np.float)
       
    # Multiply the points of the virtual space, to convert them into
    # real image points
    a = np.dot(A, a)
    b = np.dot(A, b)
    c = np.dot(A, c)
    d = np.dot(A, d)
    centro = np.dot(A, centro)
    
    areal = (int(a[0]/a[2]), int(a[1]/b[2]))
    breal = (int(b[0]/b[2]), int(b[1]/b[2]))
    creal = (int(c[0]/c[2]), int(c[1]/c[2]))
    dreal = (int(d[0]/d[2]), int(d[1]/d[2]))
    centroreal = (int(centro[0]/centro[2]), int(centro[1]/centro[2]))
    
    #The polygon and the file name of the 
    #image are painted in the center of the polygon
    points = np.array([areal, breal, creal, dreal], np.int32)
    cv2.polylines(imgout, np.int32([points]),1, (255,255,255), thickness=2)
    
    # !!!! comment 
    '''
    try:
		utilscv.draw_str(imgout, centroreal, bestImage.nameFile.upper())
	except:
		print("Trouble with darw_str")
	'''
    cv2.imshow('ImageDetector', bestImage.imageBinary)
