import pre_processors
import controlled_degradations
import edge_detectors
import cv2

# purpose of project is to evaluate entire dataset

def calculate(dataset, preprocessor, degradation, edge_detector):
    return "Testing Results as string"

def iou(edgeimg, annotatedimg, type):
    origimg = cv2.imread(edgeimg)
    ####

    ####
    # temp
    newimg = origimg
    if type == "disp":
        cv2.imwrite('iou.png', newimg)
    return "IoU Score, used for calculation results"