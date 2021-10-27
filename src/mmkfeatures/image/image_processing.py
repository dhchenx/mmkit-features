import numpy as np
from scipy import misc
import random
import skimage
import skimage.io
import skimage.transform
import cv2
from matplotlib.image import imread,imsave
from PIL import Image

class ImageUtils:
	def __init__(self):
		pass

	def load_image_array(self,image_file, image_size):
		img = skimage.io.imread(image_file)
		# GRAYSCALE
		if len(img.shape) == 2:
			img_new = np.ndarray((img.shape[0], img.shape[1], 3), dtype='uint8')
			img_new[:, :, 0] = img
			img_new[:, :, 1] = img
			img_new[:, :, 2] = img
			img = img_new

		img_resized = skimage.transform.resize(img, (image_size, image_size))

		# FLIP HORIZONTAL WIRH A PROBABILITY 0.5
		if random.random() > 0.5:
			img_resized = np.fliplr(img_resized)

		return img_resized.astype('float32')

	def get_numpy_array_rgb(self,image_file):
		img = imread(image_file)
		return img

	def save_numpy_array_rgb(self,image_file,img_data):
		imsave(image_file,img_data)

	def resize_image(self,image_file,new_size,output_file=None,format="JPEG"):
		im = Image.open(image_file)
		im.thumbnail(new_size, Image.ANTIALIAS)
		if output_file!=None:
			im.save(output_file, format)
		return np.array(im)


