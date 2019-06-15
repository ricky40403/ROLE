from __future__ import absolute_import

import os
from raindrop.dropgenerator import generateDrops
from raindrop.config import cfg
import numpy as np
import cv2

def main():
	
	

	image_folder_path = "Images"
	outputimg_folder_path = "Output_image"
	outputlabel_folder_path = "Output_label"
	for file_name in os.listdir(image_folder_path):
		image_path = os.path.join(image_folder_path, file_name)
		output_image, output_label = generateDrops(image_path, cfg)
		save_path = os.path.join(outputimg_folder_path, file_name)
		output_image.save(save_path)
		save_path = os.path.join(outputlabel_folder_path, file_name)
		output_label = np.array(output_label)
		cv2.imwrite(save_path, output_label*255)	

if __name__ == "__main__":
	main()

