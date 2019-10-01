from __future__ import absolute_import

import os
from raindrop.dropgenerator import generateDrops
from raindrop.config import cfg

from PIL import Image
import numpy as np
import cv2

def main():
	
	

	image_folder_path = "Images"
	outputimg_folder_path = "Output_image"
	outputlabel_folder_path = "Output_label"
	# using predifined label
	# input_label = Image.open("test.png")
	for file_name in os.listdir(image_folder_path):
		image_path = os.path.join(image_folder_path, file_name)
		# output image and output label is both in PIL format
		
		# randan generate by default
		output_image, output_label = generateDrops(image_path, cfg)
		# use label
		# output_image, output_label = generateDrops(image_path, cfg, inputLabel = input_label)
		
		save_path = os.path.join(outputimg_folder_path, file_name)
		# save image
		output_image.save(save_path)
		save_path = os.path.join(outputlabel_folder_path, file_name)
		# To show, need to multiply 255 
		output_label = np.array(output_label)
		cv2.imwrite(save_path, output_label*255)

if __name__ == "__main__":
	main()

