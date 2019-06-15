from __future__ import absolute_import

import os
from raindrop.dropgenerator import generateDrops

def main():
	image_folder_path = "Images"
	output_folder_path = "Output"
	for file_name in os.listdir(image_folder_path):
		image_path = os.path.join(image_folder_path, file_name)
		output_image = generateDrops(image_path)
		save_path = os.path.join(output_folder_path, file_name)
		output_image.save(save_path)


if __name__ == "__main__":
	main()
