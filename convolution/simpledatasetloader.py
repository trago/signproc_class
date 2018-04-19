# import the necessary packages
import numpy as np
import cv2
import os
import glob

class SimpleDatasetLoader(object):
    def __init__(self, preprocessors=None):
        super(SimpleDatasetLoader, self).__init__()
        # store the image preprocessor
        self.preprocessors = preprocessors

        # if the preprocessors are None, initialize them as an
        # empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePathX, imagePathY, verbose=-1, im_ext = 'jpg'):
        # initialize the list of features and labels
        i = 0
        dataX = []
        dataY = []

        # loop over the input images
        wildcard = '*.' + im_ext
        files = os.path.join(imagePathX, wildcard)
        for image_x in glob.glob(files):
            i = i + 1
            Ix = cv2.imread(image_x, cv2.IMREAD_GRAYSCALE)
            image_path, image_name = os.path.split(image_x)
            image_y = os.path.join(imagePathY, image_name)
            Iy = cv2.imread(image_y, cv2.IMREAD_GRAYSCALE)
            Ix = Ix.reshape((Ix.shape[0], Ix.shape[1], 1))/255
            Iy = Iy.reshape((Ix.shape[0], Ix.shape[1], 1))/255

            if Iy is None:
                e = IOError("Data samples are corrupted")
                raise e

            # check to see if our preprocessors are not None
            if self.preprocessors is not None:
                # loop over the preprocessors and apply each to
                # the image
                for p in self.preprocessors:
                    image = p.preprocess(image)

            # treat our processed image as a "feature vector"
            # by updating the data list followed by the labels
            dataX.append(Ix)
            dataY.append(Iy)

            # show an update every `verbose` images
            if verbose > 0 and i > 0:
                print("[INFO] Image x: {}, y: {}".format(image_x,
                    image_y))

        # return a tuple of the data and labels
        return (np.array(dataX), np.array(dataY))