import cv2
## Sobel
def sobel(imgpath):
    origimg = cv2.imread(imgpath)
    ####

    ####
    # temp
    newimg = origimg
    cv2.imwrite('edge.png', newimg)
## Prewitt
def prewitt(imgpath):
    origimg = cv2.imread(imgpath)
    ####

    ####
    # temp
    newimg = origimg
    cv2.imwrite('edge.png', newimg)
## Laplacian of Gaussian
def log(imgpath):
    origimg = cv2.imread(imgpath)
    ####

    ####
    # temp
    newimg = origimg
    cv2.imwrite('edge.png', newimg)
## Canny Edge Detection
def canny(imgpath):
    origimg = cv2.imread(imgpath)
    ####

    ####
    # temp
    newimg = origimg
    cv2.imwrite('edge.png', newimg)