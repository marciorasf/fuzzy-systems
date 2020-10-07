import pandas as pd
import cv2


def readImage(src):
    return cv2.imread(src)

def rescaleImage(image, scaleRatio):
    width = int(image.shape[1] * scaleRatio)
    height = int(image.shape[0] * scaleRatio)

    rescaledImage = cv2.resize(image, (width, height))

    return rescaledImage 

def writeImage(dest, image):
    cv2.imwrite(dest, image)