import os
import numpy as np
from PIL import Image
import pdb
import cv2
class Mosaic_Canvas(object):
	def __init__(self,patch_size=256, n=100, downscale=4, n_per_row=10, bg_color=(0,0,0), alpha=-1):
		self.patch_size = patch_size
		self.downscaled_patch_size = int(np.ceil(patch_size/downscale))
		self.n_rows = int(np.ceil(n / n_per_row))
		self.n_cols = n_per_row
		w = self.n_cols * self.downscaled_patch_size
		h = self.n_rows * self.downscaled_patch_size
		if alpha < 0:
			canvas = Image.new(size=(w,h), mode="RGB", color=bg_color)
		else:
			canvas = Image.new(size=(w,h), mode="RGBA", color=bg_color + (int(255 * alpha),))
		
		self.canvas = canvas
		self.dimensions = np.array([w, h])
		self.reset_coord()

	def reset_coord(self):
		self.coord = np.array([0, 0])

	def increment_coord(self):
		#print('current coord: {} x {} / {} x {}'.format(self.coord[0], self.coord[1], self.dimensions[0], self.dimensions[1]))
		assert np.all(self.coord<=self.dimensions)
		if self.coord[0] + self.downscaled_patch_size <=self.dimensions[0] - self.downscaled_patch_size:
			self.coord[0]+=self.downscaled_patch_size
		else:
			self.coord[0] = 0 
			self.coord[1]+=self.downscaled_patch_size
		

	def save(self, save_path, **kwargs):
		self.canvas.save(save_path, **kwargs)

	def paste_patch(self, patch):
		assert patch.size[0] == self.patch_size
		assert patch.size[1] == self.patch_size
		self.canvas.paste(patch.resize(tuple([self.downscaled_patch_size, self.downscaled_patch_size])), tuple(self.coord))
		self.increment_coord()

	def get_painting(self):
		return self.canvas

class Contour_Checking_fn(object):
	# Defining __call__ method 
	def __call__(self, pt): 
		raise NotImplementedError

class isInContourV1(Contour_Checking_fn):
	def __init__(self, contour):
		self.cont = contour

	def __call__(self, pt): 
		return 1 if cv2.pointPolygonTest(self.cont, pt, False) >= 0 else 0

class isInContourV2(Contour_Checking_fn):
	def __init__(self, contour, patch_size):
		self.cont = contour
		self.patch_size = patch_size

	def __call__(self, pt): 
		return 1 if cv2.pointPolygonTest(self.cont, (pt[0]+self.patch_size//2, pt[1]+self.patch_size//2), False) >= 0 else 0

# WholeSlideImage.pyの_getPatchGeneratorメソッドや、process_contourメソッドから呼び出される。
# 輪郭のチェック用メソッド
# Easy version of 4pt contour checking function - 1 of 4 points need to be in the contour for test to pass
class isInContourV3_Easy(Contour_Checking_fn):
	def __init__(self, contour, patch_size, center_shift=0.5):
		self.cont = contour
		self.patch_size = patch_size
		# self.shift = パッチサイズを2で割って小数点以下を切り捨てた数 x center_shift(default=0.5)
		# 例: patch_size = 512の場合、self.shift = 128.0
		self.shift = int(patch_size//2*center_shift)
	def __call__(self, pt):
		# center = パッチの中央座標 
		# center = (パッチstart x座標 + パッチサイズの半分, パッチstart y座標 + パッチサイズの半分)
		center = (pt[0]+self.patch_size//2, pt[1]+self.patch_size//2)
		# self.shiftが0より大きい場合、
		# all_points = シフトした場合の中央座標からシフト分の周囲四角形の四隅の座標
		if self.shift > 0:
			all_points = [(center[0]-self.shift, center[1]-self.shift),
						  (center[0]+self.shift, center[1]+self.shift),
						  (center[0]+self.shift, center[1]-self.shift),
						  (center[0]-self.shift, center[1]+self.shift)
						  ]
		# self.shiftが0より小さい場合、
		# all_points = [中央座標]
		else:
			all_points = [center]
		# 四隅の座標をそれぞれ抽出し、pointPolygonTestメソッドで、
		# 輪郭内にこの点が入っている場合は1を返す。
		for points in all_points:
			if cv2.pointPolygonTest(self.cont, points, False) >= 0:
				return 1
		return 0

# Hard version of 4pt contour checking function - all 4 points need to be in the contour for test to pass
class isInContourV3_Hard(Contour_Checking_fn):
	def __init__(self, contour, patch_size, center_shift=0.5):
		self.cont = contour
		self.patch_size = patch_size
		self.shift = int(patch_size//2*center_shift)
	def __call__(self, pt): 
		center = (pt[0]+self.patch_size//2, pt[1]+self.patch_size//2)
		if self.shift > 0:
			all_points = [(center[0]-self.shift, center[1]-self.shift),
						  (center[0]+self.shift, center[1]+self.shift),
						  (center[0]+self.shift, center[1]-self.shift),
						  (center[0]-self.shift, center[1]+self.shift)
						  ]
		else:
			all_points = [center]
		
		for points in all_points:
			if cv2.pointPolygonTest(self.cont, points, False) < 0:
				return 0
		return 1



		