import random
import cv2
import math
import numpy as np
import pyblur

from PIL import Image
from PIL import ImageEnhance

from raindrop import raindrop
"""
This script generate the Drop on the images
Author: Chia-Tse, Chang

"""


def CheckCollision(DropList):
	"""
	This function handle the collision of the drops
	"""
	
	listFinalDrops = []
	Checked_list = []
	list_len = len(DropList)
	# because latter raindrops in raindrop list should has more colision information
	# so reverse list	
	DropList.reverse()
	drop_key = 1
	for drop in DropList:
		# if the drop has not been handle	
		if drop.getKey() not in Checked_list:			
			# if drop has collision with other drops
			if drop.getIfColli():
				# get collision list
				collision_list = drop.getCollisionList()
				# first get radius and center to decide how  will the collision do
				final_x = drop.getCenters()[0] * drop.getRadius()
				final_y = drop.getCenters()[1]  * drop.getRadius()
				tmp_devide = drop.getRadius()
				final_R = drop.getRadius()  * drop.getRadius()
				for col_id in collision_list:				
					Checked_list.append(col_id)
					# list start from 0				
					final_x += DropList[list_len - col_id].getRadius() * DropList[list_len - col_id].getCenters()[0]
					final_y += DropList[list_len - col_id].getRadius() * DropList[list_len - col_id].getCenters()[1]
					tmp_devide += DropList[list_len - col_id].getRadius()
					final_R += DropList[list_len - col_id].getRadius() * DropList[list_len - col_id].getRadius() 
				final_x = int(round(final_x/tmp_devide))
				final_y = int(round(final_y/tmp_devide))
				final_R = int(round(math.sqrt(final_R)))
				# rebuild drop after handled the collisions
				newDrop = raindrop(drop_key, (final_x, final_y), final_R)
				drop_key = drop_key+1
				listFinalDrops.append(newDrop)
			# no collision
			else:
				drop.setKey(drop_key)
				drop_key = drop_key+1
				listFinalDrops.append(drop)			
	

	return listFinalDrops


def generateDrops(imagePath):
	"""
	This function generate the drop with random position
	"""
	maxDrop = 30
	maxR = 50
	minR = 20

	PIL_bg_img = Image.open(imagePath)
	bg_img = np.asarray(PIL_bg_img)
	# to check if collision or not
	label_map = np.zeros_like(bg_img)[:,:,0]
	imgh, imgw, _ = bg_img.shape
	
	# random drops position
	ran_pos = [(int(random.random() * imgw), int(random.random() * imgh)) for _ in range(maxDrop)]

	listRainDrops = []
	#########################
	# Create Raindrop
	#########################
	for key, pos in enumerate(ran_pos):
		# label should start from 1
		key = key+1
		radius = random.randint(minR, maxR)
		drop = raindrop(key, pos, radius)
		listRainDrops.append(drop)
		
	#########################
	# Handle Collision
	#########################
	
	collisionNum = len(listRainDrops)
	listFinalDrops = list(listRainDrops)
	loop = 0
	while collisionNum > 0:		
		loop = loop+1
		listFinalDrops = list(listFinalDrops)
		collisionNum = len(listFinalDrops)
		label_map = np.zeros_like(label_map)
		# Check Collision
		for drop in listFinalDrops:		
			# check the bounding 
			(ix, iy) = drop.getCenters()
			radius = drop.getRadius()
			ROI_WL = 2*radius
			ROI_WR = 2*radius
			ROI_HU = 3*radius
			ROI_HD = 2*radius
			if (iy-3*radius) <0 :
				ROI_HU = iy	
			if (iy+2*radius)>imgh:
				ROI_HD = imgh - iy
			if (ix-2*radius)<0:
				ROI_WL = ix			
			if  (ix+2*radius) > imgw:
				ROI_WR = imgw - ix

			# apply raindrop label map to Image's label map
			drop_label = drop.getLabelMap()
			# check if center has already has drops
			if (label_map[iy, ix] > 0):
				col_ids = np.unique(label_map[iy - ROI_HU:iy + ROI_HD, ix - ROI_WL: ix+ROI_WR])
				col_ids = col_ids[col_ids!=0]
				drop.setCollision(True, col_ids)
				label_map[iy - ROI_HU:iy + ROI_HD, ix - ROI_WL: ix+ROI_WR] = drop_label[3*radius - ROI_HU:3*radius + ROI_HD, 2*radius - ROI_WL: 2*radius+ROI_WR] * drop.getKey()
			else:				
				label_map[iy - ROI_HU:iy + ROI_HD, ix - ROI_WL: ix+ROI_WR] = drop_label[3*radius - ROI_HU:3*radius + ROI_HD, 2*radius - ROI_WL: 2*radius+ROI_WR] * drop.getKey()
				# no collision 
				collisionNum = collisionNum-1


		if collisionNum > 0:
			listFinalDrops = CheckCollision(listFinalDrops)	
			


	# add alpha for the edge of the drops
	alpha_map = np.zeros_like(label_map).astype(np.float64)
	for drop in listFinalDrops:
		(ix, iy) = drop.getCenters()
		radius = drop.getRadius()
		ROI_WL = 2*radius
		ROI_WR = 2*radius
		ROI_HU = 3*radius
		ROI_HD = 2*radius
		if (iy-3*radius) <0 :
			ROI_HU = iy	
		if (iy+2*radius)>imgh:
			ROI_HD = imgh - iy
		if (ix-2*radius)<0:
			ROI_WL = ix			
		if  (ix+2*radius) > imgw:
			ROI_WR = imgw - ix

		drop_alpha = drop.getAlphaMap()	

		alpha_map[iy - ROI_HU:iy + ROI_HD, ix - ROI_WL: ix+ROI_WR] += drop_alpha[3*radius - ROI_HU:3*radius + ROI_HD, 2*radius - ROI_WL: 2*radius+ROI_WR]
		alpha_map = alpha_map/np.max(alpha_map)*255.0
	alpha_map = cv2.GaussianBlur(alpha_map, (3, 3), 0)




	PIL_bg_img = Image.open(imagePath)
	for drop in listFinalDrops:
		# check bounding
		(ix, iy) = drop.getCenters()
		radius = drop.getRadius()		
		ROIU = iy - 3*radius
		ROID = iy + 2*radius
		ROIL = ix - 2*radius
		ROIR = ix + 2*radius
		if (iy-3*radius) <0 :
			ROIU = 0
			ROID = 5*radius		
		if (iy+2*radius)>imgh:
			ROIU = imgh - 5*radius
			ROID = imgh
		if (ix-2*radius)<0:
			ROIL = 0
			ROIR = 4*radius
		if  (ix+2*radius) > imgw:		
			ROIL = imgw - 4*radius
			ROIR = imgw


		tmp_bg = bg_img[ROIU:ROID, ROIL:ROIR,:]
		drop.updateTexture(tmp_bg)
		tmp_alpha_map  = alpha_map[ROIU:ROID, ROIL:ROIR]

		
		output = drop.getTexture()
		tmp_output = np.asarray(output).astype(np.float)[:,:,-1]
		tmp_alpha_map = tmp_alpha_map * (tmp_output/255)
		tmp_alpha_map  = Image.fromarray(tmp_alpha_map.astype('uint8'))		

		edge = ImageEnhance.Brightness(output)
		brightness = 0.3
		edge = edge.enhance(brightness)
		PIL_bg_img.paste(edge, (ix-2*radius, iy-3*radius), tmp_alpha_map)
		PIL_bg_img.paste(output, (ix-2*radius, iy-3*radius), output)
	return PIL_bg_img


