# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.nn.conv import MiniVGGNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

#grab the list of images that we'll be describing, then extract
#the class label names form the image paths
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args['dataset']))
print(len(imagePaths))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]
print(classNames)

# initialize the image preprocessors
aap = AspectAwarePreprocessor(64, 64)
