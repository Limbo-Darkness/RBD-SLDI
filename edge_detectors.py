import cv2
import numpy as np

# Each detector has two entry points:
#   - detect(imgpath)   reads from disk, writes a displayable edge.png (0-255 magnitude
#                       for gradient methods, 0/255 binary for Canny), used by the GUI
#   - detect_arr(img)   accepts a numpy array (BGR uint8), returns a float32 gradient
#                       magnitude normalised to [0,255] (or 0/255 binary for Canny)
#                       with NO disk I/O — used by batch scoring
#
# Gradient methods (Sobel, Prewitt, LoG) return the continuous magnitude so that
# result.iou() can apply a band-restricted Otsu threshold that adapts to each image's
# actual gradient distribution near the lesion boundary, rather than committing to a
# global threshold that may fire on internal texture instead of the boundary.
# Canny is already self-thresholding (NMS + hysteresis) so it returns a binary map.


## Sobel
def sobel_arr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # CV_64F to capture both positive and negative gradients before combining
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    # Return normalised float32 magnitude; iou() will threshold adaptively
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)

def sobel(imgpath):
    mag = sobel_arr(cv2.imread(imgpath))
    # Write uint8 for display — a simple midpoint threshold looks good visually
    cv2.imwrite('edge.png', mag.astype(np.uint8))


## Prewitt
def prewitt_arr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    kernelx = np.array([[-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]], dtype=np.float64)
    kernely = np.array([[-1, -1, -1],
                        [ 0,  0,  0],
                        [ 1,  1,  1]], dtype=np.float64)
    prewittx = cv2.filter2D(gray, -1, kernelx)
    prewitty = cv2.filter2D(gray, -1, kernely)
    magnitude = np.sqrt(prewittx**2 + prewitty**2)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)

def prewitt(imgpath):
    mag = prewitt_arr(cv2.imread(imgpath))
    cv2.imwrite('edge.png', mag.astype(np.uint8))


## Laplacian of Gaussian
def log_arr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Gaussian blur first to suppress noise before the Laplacian amplifies it
    blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    # Absolute value so both sides of zero-crossings appear as edges
    magnitude = np.abs(laplacian)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)

def log(imgpath):
    mag = log_arr(cv2.imread(imgpath))
    cv2.imwrite('edge.png', mag.astype(np.uint8))


## Canny — already self-thresholding via NMS + hysteresis, returns binary 0/255
def canny_arr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Otsu's threshold on the grayscale image drives the hysteresis thresholds,
    # making Canny adaptive to each image's actual contrast
    otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low  = otsu_thresh * 0.5
    high = otsu_thresh
    return cv2.Canny(gray, low, high).astype(np.float32)

def canny(imgpath):
    edge = canny_arr(cv2.imread(imgpath))
    cv2.imwrite('edge.png', edge.astype(np.uint8))
