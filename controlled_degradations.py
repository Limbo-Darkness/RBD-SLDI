import cv2
import numpy as np
 
# Each degradation has two entry points:
#   - degrade(imgpath)      reads from disk, writes 'degraded.png' - used by the GUI display path
#   - degrade_arr(img)      accepts and returns a numpy array (BGR uint8) - used by batch evaluation
 
## Gaussian Noise
def gaussNoise_arr(img):
    # Generate Gaussian noise with mean=0 and std=25, matching image shape
    noise = np.random.normal(0, 25, img.shape).astype(np.float32)
    # Add noise to image, then clip to valid [0, 255] range and convert back to uint8
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
 
def gaussNoise(imgpath):
    newimg = gaussNoise_arr(cv2.imread(imgpath))
    cv2.imwrite('degraded.png', newimg)
 
 
## Salt and Pepper Noise
def saltpepper_arr(img):
    newimg = img.copy()
    # Apply salt (white) and pepper (black) noise at 2% density each
    density = 0.02
    h, w = img.shape[:2]
    num_pixels = int(density * h * w)
    # Salt: set random pixels to 255
    salt_coords = (np.random.randint(0, h, num_pixels), np.random.randint(0, w, num_pixels))
    newimg[salt_coords] = 255
    # Pepper: set random pixels to 0
    pepper_coords = (np.random.randint(0, h, num_pixels), np.random.randint(0, w, num_pixels))
    newimg[pepper_coords] = 0
    return newimg
 
def saltpepper(imgpath):
    newimg = saltpepper_arr(cv2.imread(imgpath))
    cv2.imwrite('degraded.png', newimg)
 
 
## Blur
def blur_arr(img):
    # Apply Gaussian blur with a 15x15 kernel (strong enough to be visually meaningful)
    return cv2.GaussianBlur(img, (15, 15), 0)
 
def blur(imgpath):
    newimg = blur_arr(cv2.imread(imgpath))
    cv2.imwrite('degraded.png', newimg)
 
 
## Reduced Illumination
def debright_arr(img):
    # Convert to HSV to manipulate the Value (brightness) channel independently
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Reduce brightness by 50%
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.5, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
 
def debright(imgpath):
    newimg = debright_arr(cv2.imread(imgpath))
    cv2.imwrite('degraded.png', newimg)
 
