import cv2
### Noise Reduction
## Gaussian Smoothing
def gaussSmooth(imgpath):
    origimg = cv2.imread(imgpath)
    ####

    ####
    #temp
    newimg = origimg
    cv2.imwrite('preprocessed.png', newimg)
## Median Filtering
def medianFilter(imgpath):
    origimg = cv2.imread(imgpath)
    ####

    ####
    # temp
    newimg = origimg
    cv2.imwrite('preprocessed.png', newimg)
## Bilateral Filtering
def bilateralFilter(imgpath):
    origimg = cv2.imread(imgpath)
    ####

    ####
    # temp
    newimg = origimg
    cv2.imwrite('preprocessed.png', newimg)
### Contrast Enhancement
## Histogram Equalization
def histogramEqual(imgpath):
    origimg = cv2.imread(imgpath)
    ####

    ####
    # temp
    newimg = origimg
    cv2.imwrite('preprocessed.png', newimg)
## CLAHE Contrast Limited Adaptive Histogram Equalization
def CLAHE(imgpath):
    origimg = cv2.imread(imgpath)
    ####

    ####
    # temp
    newimg = origimg
    cv2.imwrite('preprocessed.png', newimg)