import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os.path
from scipy.ndimage.measurements import label
from collections import deque
from lesson_functions import *
from utils import *

if os.path.isfile("svc_pickle.p") == False :
  from save_svc import *
  save_svc()

dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

history = deque(maxlen = 8)

def process_img(img):
  ystart = 400
  ystop = 656
  
  scale = 1.0
  windows_img, box_list_10 = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

  scale = 1.5
  windows_img, box_list_15 = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

  scale = 2.0
  windows_img, box_list_20 = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

  box_list = []
  for b in box_list_10:
    box_list.append(b)
  for b in box_list_15:
    box_list.append(b)
  for b in box_list_20:
    box_list.append(b)

  heat = np.zeros_like(img[:,:,0]).astype(np.float)
  heat = add_heat(heat,box_list)
  heat = apply_threshold(heat,1)
  heatmap_img = np.clip(heat, 0, 255)

  history.append(heatmap_img)
  avg_heatmap_img = np.zeros_like(heatmap_img)
  for heatmap_history in history:
    avg_heatmap_img = avg_heatmap_img + heatmap_history
  avg_heatmap_img = avg_heatmap_img / len(history)

  labels = label(avg_heatmap_img)
  cars_img = draw_labeled_bboxes(np.copy(img), labels)
  return windows_img, heatmap_img, cars_img

out_images = []
out_titles = []
images = glob.glob('test_images/test*.jpg')
for filename in images:
  img = mpimg.imread(filename)

  history = deque(maxlen = 8)
  windows_img, heatmap_img, cars_img = process_img(img)

  out_images.append(img)
  out_images.append(windows_img)
  out_images.append(heatmap_img)
  out_images.append(cars_img)

  out_titles.append(filename + '- Original')
  out_titles.append(filename + '- Windows')
  out_titles.append(filename + '- Heat Map')
  out_titles.append(filename + '- Car Positions')

export_images(out_images, out_titles, '02_result.png', figsize=(24,24), columns=4)

def pipeline(img):
  windows_img, heatmap_img, cars_img = process_img(img)
  return cars_img
  
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

history = deque(maxlen = 8)
video_output = 'test_video_output.mp4'
clip1 = VideoFileClip("test_video.mp4")
output_clip = clip1.fl_image(pipeline)
output_clip.write_videofile(video_output, audio=False)

history = deque(maxlen = 8)
video_output = 'project_video_output.mp4'
clip1 = VideoFileClip("project_video.mp4")
output_clip = clip1.fl_image(pipeline)
output_clip.write_videofile(video_output, audio=False)
