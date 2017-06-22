import matplotlib.image as mpimg
import numpy as np
import cv2
from utils import *
from hog import *

# Get list of filename for images
cars, notcars = get_image_list()

# Pick indexes for car / not-car examples
car_ind = np.random.randint(0, len(cars))
notcar_ind = np.random.randint(0, len(notcars))

# Load images
car_image = mpimg.imread(cars[car_ind])
notcar_image = mpimg.imread(notcars[notcar_ind])

export_images([car_image, notcar_image], ['Car', 'Not-Car'], 
    '00_car_not_car.png')

# Define HOG parameters
orient = 9
pix_per_cell = 8
cell_per_block = 2

# Convert to gray 
car_gray = cv2.cvtColor(car_image, cv2.COLOR_RGB2GRAY)
notcar_gray = cv2.cvtColor(notcar_image, cv2.COLOR_RGB2GRAY)

## Call our function with vis=True to see an image output
car_features, car_hog_image = get_hog_features(car_gray, orient, 
    pix_per_cell, cell_per_block, 
    vis=True, feature_vec=False)
notcar_features, notcar_hog_image = get_hog_features(notcar_gray, orient, 
    pix_per_cell, cell_per_block, 
    vis=True, feature_vec=False)

export_images([car_gray, car_hog_image, notcar_gray, notcar_hog_image], 
    ['Car', 'Car Hog', 'Not-Car', 'Not-Car Hog'], 
    '01_car_not_car_hog.png', columns=4, cmap='gray')

## Plot the examples
#fig = plt.figure()
#plt.subplot(121)
#plt.imshow(image, cmap='gray')
#plt.title('Example Car Image')
#plt.subplot(122)
#plt.imshow(hog_image, cmap='gray')
#plt.title('HOG Visualization')
#
#plt.savefig('a.png')
