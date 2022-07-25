from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle

from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils, models
import torch.nn.functional as F

from PIL import Image
import h5py

from random import randrange

# 本ファイル下部のWhole_Slide_Bag_FPクラスのコンストラクタ(__init__)等から呼び出されている。
def eval_transforms(pretrained=False):
	# 事前学習済の重みを用いる場合、resnet学習時と同じ前処理を行うと思われるが、imagenetを学習したモデルの前処理に標準偏差は用いただろうか？
	# 理由は理解しきれていないが、確か通常の前処理では平均値を引くだけの処理をしていた気がする。
	# 今回は平均値を引いてから標準偏差で割る、本当の標準化を行っていると思われる。
	if pretrained:
		# mean:標準化に用いるimagenet画像のピクセル値の平均値
		mean = (0.485, 0.456, 0.406)
		# std:標準化に用いるimagenet画像のピクセル値の標準偏差
		std = (0.229, 0.224, 0.225)

	# 事前学習済の重みを用いない場合、RGBorGBRで平均も標準偏差も0.5としている。
	else:
		mean = (0.5,0.5,0.5)
		std = (0.5,0.5,0.5)

	# transforms.Composeは複数のtransformsを連続で行える。
	trnsfrms_val = transforms.Compose(
					[
					 # テンソルに変換する
					 transforms.ToTensor(),
					 # 平均値を引いて、標準偏差で割る処理
					 transforms.Normalize(mean = mean, std = std)
					]
				)
	# 上記のtransfrms_valを返す。上記の処理を行う物を返すイメージか
	return trnsfrms_val

class Whole_Slide_Bag(Dataset):
	def __init__(self,
		file_path,
		pretrained=False,
		custom_transforms=None,
		target_patch_size=-1,
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.pretrained=pretrained
		if target_patch_size > 0:
			self.target_patch_size = (target_patch_size, target_patch_size)
		else:
			self.target_patch_size = None

		if not custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained)
		else:
			self.roi_transforms = custom_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['imgs']
			self.length = len(dset)

		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['imgs']
		for name, value in dset.attrs.items():
			print(name, value)

		print('pretrained:', self.pretrained)
		print('transformations:', self.roi_transforms)
		if self.target_patch_size is not None:
			print('target_size: ', self.target_patch_size)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			img = hdf5_file['imgs'][idx]
			coord = hdf5_file['coords'][idx]
		
		img = Image.fromarray(img)
		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)
		img = self.roi_transforms(img).unsqueeze(0)
		return img, coord

# extract_features_fp.pyのcompute_w_loader関数から呼び出されている。
class Whole_Slide_Bag_FP(Dataset):
	# デフォルトではpretrained=Falseだが、compute_w_loaderから呼び出すときはTrueにしているのでTrueになると思われる。
	def __init__(self,
		file_path,
		wsi,
		pretrained=False,
		custom_transforms=None,
		custom_downsample=1,
		target_patch_size=-1
		):
		"""
		Args:
			file_path (string): 入力h5ファイル
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): サンプルに適用される変形(オプション)
			custom_transforms (callable, optional): Optional transform to be applied on a sample
			custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
			target_patch_size (int): Custom defined image size before embedding
		"""
		self.pretrained=pretrained
		self.wsi = wsi
		# custom_transformsがNoneやFalseなら、本ファイル上部に定義されたeval_transforms関数を呼び出し、返り値をroi_transformsに入れる。
		# デフォルトはNoneとおもわれる。
		# この関数により、roi_transformsに、データに対する標準化処理が入っている。
		if not custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained)
		else:
		# custom_transformsがNoneでないなら、roi_transformsにcustom_transformsを入れる。
			self.roi_transforms = custom_transforms

		self.file_path = file_path

		# 入力h5ファイル(パッチファイル)を読み込んで、パッチレベル、パッチサイズ、大きさ、'coords'等の情報を取り出している。coordsはパッチ毎の左上端座標と思われる
		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
			# target_patch_sizeが1以上なら、パッチサイズを幅×高さのタプル((128,128)等)にしている。
			if target_patch_size > 0:
				self.target_patch_size = (target_patch_size, ) * 2
			# target_patch_sizeが1以上でなくて、更にcustom_downsampleが2以上ならその値でpatch_sizeを割る
			# つまり、最終のパッチサイズをtarget_patch_size引数で手動で指定するのか、それともcustom_downsaple引数で元のパッチサイズから割り算して求めるかの違い。
			# つまり、パッチの数は変えずに、１つ１つのパッチサイズを小さくしているのか？タプルを×2しているので、要素が2つに増える。つまりパッチサイズが128なら(128,128)になる。
			# これは、後でresizeするときのイメージのサイズを表すため。
			elif custom_downsample > 1:
				self.target_patch_size = (self.patch_size // custom_downsample, ) * 2
			# 上記のいずれでもない場合は,target_patch_size=Noneとなる。
			else:
				self.target_patch_size = None
		# 要確認,以下のsummaryメソッド実行と思われる。
		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		# nameにはdownsample,downsampled_level_dim等のパラメータ名、valueにはその値[1.1.]や[49920 20736]等が入っている。
		for name, value in dset.attrs.items():
			#print(f"name,value:{name}, {value}")
			print(name, value)

		print('\nfeature extraction settings')
		print('target patch size: ', self.target_patch_size)
		print('pretrained: ', self.pretrained)
		print('transformations: ', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			# 試し1行
			#print("coord",hdf5_file['coords'][idx])
			coord = hdf5_file['coords'][idx]
		# 入力wsiを、h5ファイル内のパッチレベルやサイズ等の情報を元に読み込んで、RGBに変換している。
		# coordの中には、パッチで切り分ける左上の座標が大量に入っている。座標から、パッチサイズの大きさにwsiを切り分けているのは明らか。
		# つまり、読み込みは全てパッチ毎に切り分けて読み込んでいる。その後でミニバッチに纏めている。
		img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

		# target_patch_sizeがNoneでないなら、そのタプルの値に合わせて、イメージをリサイズしている。
		# という事は、extract_features_fp.pyのtarget_patch_size引数でサイズを指定したり、ダウンサンプル引数を指定してもread_regionによる処理の重さは変わらないという事。
		# であれば、処理を軽くしたいなら最初のパッチに分けて穴部分を削除する処理の段階でダウンサンプルしておくことが効果的と思われる。
		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)
		# roi_transformsでイメージを標準化して、unsqueeze(0)で頭にバッチ次元を追加し、学習できるようにしている。
		img = self.roi_transforms(img).unsqueeze(0)
		return img, coord

class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	# ファイルのデータフレームの長さを読み込む
	# 例：extract_features_fp.pyでこのクラスのインスタンスbags_datasetを作ると、
	# len(bags_dataset)->3
	def __len__(self):
		return len(self.df)
	# インスタンスに[1]や[0]などを付けることで、slide_id列のその番号の項目を取り出せる。
	# 例：bags_dataset[1]->'Normal'
	def __getitem__(self, idx):
		return self.df['slide_id'][idx]




