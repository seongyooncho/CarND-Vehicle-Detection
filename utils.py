import matplotlib.pyplot as plt
import glob

def get_image_list():
  images = glob.glob('./data/**/*.png', recursive=True)
  cars = []
  notcars = []

  for image in images:
    if 'non-vehicles' in image:
      notcars.append(image)
    else:
      cars.append(image)

  return cars, notcars

def export_images(images, titles, filename, columns=2, cmap=None, figsize=None):
  fig = plt.figure(figsize=figsize)

  rows = (len(images) - 1) // columns + 1
  
  for index, (image, title) in enumerate(zip(images, titles)):
    plt.subplot(rows, columns, index + 1)
    plt.imshow(image, cmap=cmap)
    plt.title(title)

  plt.savefig('./output_images/' + filename)
