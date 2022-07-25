import pickle
import torch
import numpy as np
import torch.nn as nn
import pdb

import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubsetSequentialSampler(Sampler):
	"""Samples elements sequentially from a given list of indices, without replacement.

	Arguments:
		indices (sequence): a sequence of indices
	"""
	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)

def collate_MIL(batch):
	# イメージのテンソルの連結
	img = torch.cat([item[0] for item in batch], dim = 0)
	# ラベル(64bit符号付き整数)
	label = torch.LongTensor([item[1] for item in batch])
	# チェック用
	print("########################")
	print("batch_shape:",batch[0][0].shape)
	print("label:",label)
	return [img, label]

def collate_features(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	coords = np.vstack([item[1] for item in batch])
	return [img, coords]


def get_simple_loader(dataset, batch_size=1, num_workers=1):
	# eval_utils.pyのeval関数などから呼び出されている。return_splits
	# num_workers,pin_memoryはcudaの場合だけ指定し、cpuの場合はデフォルト=0(シングル実行)
	# num_workers:ミニバッチを作成する際の並列実行数　os.cpu_core()の数によっては2以上にした方が良いかも。
	# なぜnum_workersが2つあるのかは謎。(最初は4,次はデフォルトでは1の様子)
	# pin_memory:Trueにすることでautomatic memory pinningが使用できる。CPUのメモリー固定がされ、ページングされないのでその分高速化が期待できる。
	# ただし、メモリー固定されればその分の領域を処理が終了するまで占有し続けるので、PC全体のメモリのやりくりは難しくなると思われる。
	# 参考＝https://qiita.com/sugulu_Ogawa_ISID/items/62f5f7adee083d96a587#1-dataloader%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6
	kwargs = {'num_workers': 4, 'pin_memory': False, 'num_workers': num_workers} if device.type == "cuda" else {}
	# DataLoader：データとラベルを取り出すためのイテレータを返す。つまりここでは、loaderがイテレータとなる。
	# 参考＝https://qiita.com/mathlive/items/2a512831878b8018db02
	# sampler：データセットから読み込む順序を既定している。SequentialSamplerの場合はshuffle=Falseとした場合と同じで、データセットから順番に読み込むだけ。
	# collate_fn：Datasetから取得した複数のサンプルを結合して１つのミニバッチを作成する処理を行う関数
	#             collate_MIL関数が指定されているが、collate_MILの引数であるbatchが指定されていない。
	#             この場合、恐らくDataLoader内部でcollate_MILの画像とラベルを結合する処理がされると思われる。batchは内部的に指定？
	# ここでは、本ファイル上部で定義されている独自のcollate_MIL関数を呼び出し、イメージなどを結合している。
	# batch_size: １つのバッチからいくつのサンプルを取り出すか。本コードでも1と思われる。https://pytorch.org/docs/stable/data.html
	loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_MIL, **kwargs)
	return loader 

# core_utils.pyのtrain関数から、train,val,testそれぞれのスプリットから呼び出されている。
def get_split_loader(split_dataset, training = False, testing = False, weighted = False):
	"""
		return either the validation loader or training loader 
	"""
	kwargs = {'num_workers': 4} if device.type == "cuda" else {}
	if not testing:
		if training:
			if weighted:
				# split_datasetはcore_utils.pyのtrain_loader等のtrain_split等から
				weights = make_weights_for_balanced_classes_split(split_dataset)
				loader = DataLoader(split_dataset, batch_size=1, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate_MIL, **kwargs)	
			else:
				loader = DataLoader(split_dataset, batch_size=1, sampler = RandomSampler(split_dataset), collate_fn = collate_MIL, **kwargs)
		else:
			loader = DataLoader(split_dataset, batch_size=1, sampler = SequentialSampler(split_dataset), collate_fn = collate_MIL, **kwargs)
	
	else:
		ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
		loader = DataLoader(split_dataset, batch_size=1, sampler = SubsetSequentialSampler(ids), collate_fn = collate_MIL, **kwargs )

	return loader

def get_optim(model, args):
	if args.opt == "adam":
		# 最適化アルゴリズム=Adam、filter()関数でrequires_gradがTrueのものだけモデルパラメータを返す。weight_decayは重み減衰(L2正則化),lrは引数学習率
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
	elif args.opt == 'sgd':
		# 最適化アルゴリズム=SGD、filter()関数でrequires_gradがTrueのものだけモデルパラメータを返す。weight_decayは重み減衰(L2正則化),lrは引数学習率
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
	else:
		# optがadamでもsgdでもない場合は実装エラーとなる。
		raise NotImplementedError
	# Adam又はSGDのoptimizerを返す。
	return optimizer

def print_network(net):
	num_params = 0
	num_params_train = 0
	print(net)
	
	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n
	
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)


def generate_split(cls_ids, val_num, test_num, samples, n_splits = 5,
	seed = 7, label_frac = 1.0, custom_test_ids = None):
	indices = np.arange(samples).astype(int)
	
	if custom_test_ids is not None:
		indices = np.setdiff1d(indices, custom_test_ids)

	np.random.seed(seed)
	for i in range(n_splits):
		all_val_ids = []
		all_test_ids = []
		sampled_train_ids = []
		
		if custom_test_ids is not None: # pre-built test split, do not need to sample
			all_test_ids.extend(custom_test_ids)

		for c in range(len(val_num)):
			# possible_indicesは、cls_ids[c]:c=0->正常 c=1->腫瘍のself.patient_cls_idsのリスト
			# indicesはcase_idの数だけ番号を振ったリスト[0 1 2 3 ...]なのでcase_id毎の全数
			# np.intersect1dはx,yの共通項だけ残すので、case_id毎の全数の内、正常=0と腫瘍=1でそれぞれ以下を計算している。
			possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class
			# val_num[0]は正常のvalの数、val_num[1]は腫瘍のvalの数。デフォルトで前者は正常全数×0.1、後者は腫瘍全数×0.1になる。
			# val_idsはc=0では正常全数からval_num[0]数だけランダムで番号を取り出している。(val_num[0]=3なら3つランダムで取り出す)
			val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids
			# setdiff1dでval_ids(val数分)をpossible_indicesから引いた残りをremaining_idsとする。
			remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
			# 以下のextendでall_val_idsリストにval_idsリストの項目を追加している。
			all_val_ids.extend(val_ids)

			if custom_test_ids is None: # sample test split

				test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
				remaining_ids = np.setdiff1d(remaining_ids, test_ids)
				all_test_ids.extend(test_ids)

			if label_frac == 1:
				sampled_train_ids.extend(remaining_ids)
			
			else:
				sample_num  = math.ceil(len(remaining_ids) * label_frac)
				slice_ids = np.arange(sample_num)
				sampled_train_ids.extend(remaining_ids[slice_ids])

		yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
	if n is None:
		return collections.deque(iterator, maxlen=0)
	else:
		return next(islice(iterator,n, None), default)

def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

	return error

def make_weights_for_balanced_classes_split(dataset):
	N = float(len(dataset))                                           
	# チェック用t	slide_cls_idsはどこから？ datasetはget_split_loaderからその前はcore_utils.pyのtrain_splitから
	print("#dataset.slide_cls_ids:",dataset.slide_cls_ids)
	print("#dataset.slide_cls_ids[0]:",dataset.slide_cls_ids[0])
	print("#dataset.slide_cls_ids[1]:",dataset.slide_cls_ids[1])
	print("#len(dataset.slide_cls_ids):",len(dataset.slide_cls_ids))
	print("#N:",N)
	# チェック用t終わり
	weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
	weight = [0] * int(N)                                           
	for idx in range(len(dataset)):   
		y = dataset.getlabel(idx)                        
		weight[idx] = weight_per_class[y]                                  

	return torch.DoubleTensor(weight)

def initialize_weights(module):
	for m in module.modules():
		# モデルのmがnn.Linear型であれば、Xavierの初期値を使い初期化する。
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()
		
		# モデルのmがnn.BachNorm1d(バッチ正規化)型であれば、
		# 重みの初期値を1,biasの初期値を0として初期化する。
		elif isinstance(m, nn.BatchNorm1d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)

