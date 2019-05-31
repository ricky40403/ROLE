import random
import cv2
import numpy as np
import pyblur
from PIL import Image
from RainDrop import raindrop
maxDrop = 10
maxR = 50
minR = 20

PIL_bg_img = Image.open("test.jpg")
bg_img = np.asarray(PIL_bg_img)
imgh, imgw, _ = bg_img.shape
print(imgw, imgh)

ran_pos = [(int(random.random() * imgw), int(random.random() * imgh)) for _ in range(maxDrop)]

for pos in ran_pos:
	radius = random.randint(minR, maxR)
#	ran_pos = ran_pos[0]
#	ran_pos = (400, 400)
	print(pos)
	print("Radius:{}".format(radius))
	drop = raindrop(ran_pos, radius)
	(ix, iy) = pos
	#blur_bg = pyblur.GaussianBlur(Image.fromarray(np.uint8(bg_img)), 3)
	#blur_bg = np.asarray(blur_bg)

	tmp_bg = bg_img[iy-3*radius:iy+2*radius, ix-2*radius:ix+2*radius,:]
	drop.updateTexture(tmp_bg)
	output = drop.getTexture()
	label = drop.labelmap
	#cv2.imwrite("blur.bmp", blur_bg)
	cv2.imwrite("label.bmp", label)
	black = np.zeros_like(output)
	black = Image.fromarray(np.uint8(black))

	PIL_bg_img.paste(black, (ix-2*radius, iy-3*radius), output)
	PIL_bg_img.paste(output, (ix-2*radius, iy-3*radius), output)
PIL_bg_img.save("Output.bmp")


