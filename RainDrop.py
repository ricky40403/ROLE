import cv2
import math
import numpy as np
from PIL import Image
import pyblur


class raindrop():
	def __init__(self, centerxy, radius):			
		self.center = centerxy
		self.radius = radius
		self.type = "default"
		self.labelmap = np.zeros((self.radius * 5, self.radius*4))		
		self.background = None
		self.texture = None
		self._create_label()

	def setCenter(self, centerxy):
		self.center = centerxy
		self._create_label()

	def setRadius(self, radius):
		self.radius = radius
		self.labelmap = np.zeros((self.radius * 5, self.radius*4))
	
	def setBackground(self, bg):
		self.background = bg



	def updateTexture(self, bg):
		fg = pyblur.GaussianBlur(Image.fromarray(np.uint8(bg)), 5)
		fg = np.asarray(fg)

		K = np.array([[30*self.radius, 0, 2*self.radius],
				[0., 20*self.radius, 3*self.radius],
				[0., 0., 1]])

		D = np.array([0.0, 0.0, 0.0, 0.0])
		Knew = K.copy()
		Knew[(0,1), (0,1)] = math.pow(self.radius, 1/3)*2 * Knew[(0,1), (0,1)]
		fisheye = cv2.fisheye.undistortImage(fg, K, D=D, Knew=Knew)

		tmp = np.expand_dims(self.labelmap, axis = -1)
#		print(np.unique(self.labelmap))
		print(fg.shape)
		print(fisheye.shape)
		print(tmp.shape)
		tmp = np.concatenate((fisheye, tmp), axis = 2)

		self.texture = Image.fromarray(tmp.astype('uint8'), 'RGBA')


	def getTexture(self):
		return self.texture
		

	def _create_label(self):
		if self.type == "default":
			self._createDefaultDrop()
		elif self.type == "splash":
			self._createSplashDrop()

	def _createDefaultDrop(self):
		cv2.circle(self.labelmap, (self.radius * 2, self.radius * 3), self.radius, 128, -1)
#		print(self.radius)
#		print(int(math.sqrt(3) * self.radius))
		cv2.ellipse(self.labelmap, (self.radius * 2, self.radius * 3), (self.radius, int(1.3*math.sqrt(3) * self.radius)), 0, 180, 360, 128, -1)
#		self.labelmap = cv2.GaussianBlur(self.labelmap, (19,19), 0)
		self.labelmap = pyblur.GaussianBlur(Image.fromarray(np.uint8(self.labelmap)), 10)
		self.labelmap = np.asarray(self.labelmap).astype(np.float)
		self.labelmap = self.labelmap/np.max(self.labelmap)*255.0


	def _createSplashDrop(self):
		pass


	
	
