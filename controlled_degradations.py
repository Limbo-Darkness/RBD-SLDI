import cv2
## Gaussian Noise
def gaussNoise(imgpath):
    origimg = cv2.imread(imgpath)
    ####

    ####
    # temp
    newimg = origimg
    cv2.imwrite('degraded.png', newimg)
## Salt and Pepper Noise
def saltpepper(imgpath):
    origimg = cv2.imread(imgpath)
    ####

    ####
    # temp
    newimg = origimg
    cv2.imwrite('degraded.png', newimg)
## Blur
def blur(imgpath):
    origimg = cv2.imread(imgpath)
    ####

    ####
    # temp
    newimg = origimg
    cv2.imwrite('degraded.png', newimg)
## Reduced Illumination
def debright(imgpath):
    origimg = cv2.imread(imgpath)
    ####

    ####
    # temp
    newimg = origimg
    cv2.imwrite('degraded.png', newimg)