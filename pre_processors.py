import cv2
import numpy as np

# Each processor has two entry points:
#   - process(imgpath)      reads from disk, writes 'preprocessed.png' - used by the GUI display path
#   - process_arr(img)      accepts and returns a numpy array (BGR uint8) - used by batch evaluation
# The _arr variants are the canonical implementations; the path-based wrappers just
# load/save around them so all logic lives in one place.

### Noise Reduction

## Gaussian Smoothing
def gaussSmooth_arr(img):
    # 9x9 kernel provides strong noise suppression while preserving overall structure;
    # sigmaX=0 lets OpenCV auto-calculate sigma from kernel size
    return cv2.GaussianBlur(img, (9, 9), sigmaX=0)

def gaussSmooth(imgpath):
    newimg = gaussSmooth_arr(cv2.imread(imgpath))
    cv2.imwrite('preprocessed.png', newimg)


## Median Filtering
def medianFilter_arr(img):
    # Kernel size 7 is effective at removing salt-and-pepper noise while
    # keeping edges sharper than Gaussian smoothing
    return cv2.medianBlur(img, 7)

def medianFilter(imgpath):
    newimg = medianFilter_arr(cv2.imread(imgpath))
    cv2.imwrite('preprocessed.png', newimg)


## Bilateral Filtering
def bilateralFilter_arr(img):
    # d=9: neighbourhood diameter; sigmaColor=75: how much color difference is tolerated;
    # sigmaSpace=75: spatial reach - together these smooth noise while preserving edges
    return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

def bilateralFilter(imgpath):
    newimg = bilateralFilter_arr(cv2.imread(imgpath))
    cv2.imwrite('preprocessed.png', newimg)


### Contrast Enhancement

## Histogram Equalization
def histogramEqual_arr(img):
    # Equalize in YCrCb space rather than BGR to avoid hue shifts:
    # only the luma (Y) channel is equalized, preserving natural skin tones
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def histogramEqual(imgpath):
    newimg = histogramEqual_arr(cv2.imread(imgpath))
    cv2.imwrite('preprocessed.png', newimg)


## CLAHE — Contrast Limited Adaptive Histogram Equalization
def CLAHE_arr(img):
    # clipLimit=2.0 caps amplification to reduce noise boosting;
    # tileGridSize=(8,8) divides the image into local regions for adaptive equalization -
    # better suited to dermoscopic images where lighting varies across the frame
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def CLAHE(imgpath):
    newimg = CLAHE_arr(cv2.imread(imgpath))
    cv2.imwrite('preprocessed.png', newimg)
