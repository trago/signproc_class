import cv2
import os
import glob

def create_sintetic_samples(imgs_path, out_path, im_ext = 'jpg'):
    wildcard = '*.' + im_ext
    files = os.path.join(imgs_path, wildcard)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for image_x in glob.glob(files):
        if os.path.isfile(image_x):
            image_path, image_name = os.path.split(image_x)
            image_y = os.path.join(out_path, image_name)
            print("Processing file with Sobel: {0} -> {1} ...".format(image_x, image_y))
            Ix = cv2.imread(image_x, cv2.IMREAD_GRAYSCALE)
            Iy = cv2.Canny(Ix, 12, 3)
            #Iy = cv2.Laplacian(Ix, cv2.CV_8U)
            #Iy = cv2.blur(Ix, (1,15))
            #Iy = cv2.normalize(Iy, 1, 0, cv2.NORM_MINMAX)

            cv2.imwrite(image_y, Iy)

create_sintetic_samples('../SMILEsmileD/SMILEs/positives/positives7', '../SMILEsmileD/SMILEs/positives/output')

