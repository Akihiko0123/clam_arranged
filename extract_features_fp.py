import torch
import torch.nn as nn
from math import floor
import os
import random
import numpy as np
import pdb
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
from models.resnet_custom import resnet50_baseline
import argparse
from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
import openslide

# cudaが使えるならtorch.deviceをcudaに、使えないならcpuにする。
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 本ファイルの下の方の処理から呼び出されている。pretrainedは引数指定はないので、基本的に事前学習済みの重みが使用される。
def compute_w_loader(file_path, output_path, wsi, model,
 	batch_size = 8, verbose = 0, print_every=20, pretrained=True, 
	custom_downsample=1, target_patch_size=-1):
	"""
	args:
		file_path: 入力h5ファイルが入ったフォルダパス
		file_path: directory of bag (.h5 file)
		output_path: 計算した特徴(h5ファイル)を保存するフォルダ
		output_path: directory to save computed features (.h5 file)
		model: pytorchモデル
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: (恐らく)処理表示をどの程度詳しくするか
		verbose: level of feedback
		pretrained: imagenetで事前学習済みの重みを使う
		pretrained: use weights pretrained on imagenet
		custom_downsample: 画像パッチのダウンサンプリングのレベル
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: 埋め込み前のカスタマイズで変更された画像のサイズ
		target_patch_size: custom defined, rescaled image size before embedding
	"""
	# dataset_h5.pyのWhole_Slide_Bag_FPクラスのインスタンスがdatasetになる。
	# ここでは、
	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, 
		custom_downsample=custom_downsample, target_patch_size=target_patch_size)
	# 角括弧dataset[]はWhole_Slide_Bag_FPクラスの__getitem__メソッドを呼び出す処理。
	# xには標準化して学習の準備ができたイメージが入る。
	# yにはh5ファイルの['coords'][0]に格納されたデータ？が入る。
	x, y = dataset[0]
	kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
	# データを１バッチずつ？取り出すローダー
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

	# varboseが1以上なら処理中のファイル名と、合計バッチ数を表示する。(NCCやNormal等が6とか4とかなので、切り分けが大きい？)
	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'
	# loaderからbatchと左上端座標を取り出して繰り返す。count=繰り返し番号
	for count, (batch, coords) in enumerate(loader):
		with torch.no_grad():	
			# print_every==20なので、20バッチ毎に処理がどこまで進んだか表示させると思われる。
			if count % print_every == 0:
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
			batch = batch.to(device, non_blocking=True)
			# チェック用001
#			print("#batch:",batch)
			# チェック用001終わり
			# resnetのモデルにbatch番号毎のデータを入れて、出力をnumpy配列のfeaturesに入れる。
			features = model(batch)
			features = features.cpu().numpy()

			# featuresとバッチの左上端座標をasset_dictに入れる。
			asset_dict = {'features': features, 'coords': coords}
			# file_utils.pyのsave_hdf5関数を呼び出してh5ファイルとして保存
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path


# １．引数を読み込む
parser = argparse.ArgumentParser(description='Feature Extraction')
# create_patches_fp.pyで作成したディレクトリ(h5ファイルが入ったディレクトリ"patches"の親ディレクトリ)
parser.add_argument('--data_h5_dir', type=str, default=None)
# wsiの入ったディレクトリ
parser.add_argument('--data_slide_dir', type=str, default=None)
# スライドの拡張子
parser.add_argument('--slide_ext', type=str, default= '.svs')
# 処理するファイル名が拡張子抜きで書かれてたcsvファイル
parser.add_argument('--csv_path', type=str, default=None)
# 出力先フォルダパス
parser.add_argument('--feat_dir', type=str, default=None)
# バッチサイズ
parser.add_argument('--batch_size', type=int, default=256)
# Trueの場合は、出力先に同じ名前の.ptファイルがあっても上書きする。Falseの場合は処理をスキップする。
parser.add_argument('--no_auto_skip', default=False, action='store_true')
# ダウンサンプリング
parser.add_argument('--custom_downsample', type=int, default=1)
# ターゲットパッチサイズ
parser.add_argument('--target_patch_size', type=int, default=-1)
args = parser.parse_args()

# ２．import等ではなく、ファイル名を指定して実行されたら起動
if __name__ == '__main__':
	# データセットを初期化している旨記載
	print('initializing dataset')
	# csv_pathをチェックして、ない場合は実装エラー
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError
	# datasets/dataset_h5.pyファイルのDataset_All_Bagsクラスのインスタンスbags_datasetを作成
	# len(bags_dataset)でcsvのslide_id数が出てきて、bags_dataset[1]で"Normal"が出てくるなど
	# csv内のスライド名の取り出しが便利になる。
	bags_dataset = Dataset_All_Bags(csv_path)
	# 出力フォルダ作成
	os.makedirs(args.feat_dir, exist_ok=True)
	# 出力フォルダ内にpt_filesフォルダ作成
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	# 出力フォルダ内にh5_filesフォルダ作成
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	# pt_files内のファイルをリストにしたものをdest_filesに代入する。
	# この時点でファイルはないのでは
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	print('loading model checkpoint')
	# resnet_custom.pyのresnet50_baseline関数を呼び出す
	model = resnet50_baseline(pretrained=True)
	# modelをcudaかcpuに繋げる
	model = model.to(device)
	
	# print_network(model)
	# cudaが1個より多いならGPUを並列に使う。
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		
	# evalモードなので、dropoutやbattch normがonになっている？
	# 参考：https://qiita.com/tatsuya11bbs/items/86141fe3ca35bdae7338
	model.eval()
	# 用いるデータの数
	total = len(bags_dataset)

	# データ(スライド名)の数だけ繰り返す
	for bag_candidate_idx in range(total):
		# slide_idは、スライド名の拡張子を除いた部分(Normal,NCC等)
		# ということは、READMEには拡張子を外すだけでもいいとあったが、拡張子すら付けたままでも良かったと思われる。
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		# bag_nameにslide_id+.h5つまりNCC.h5等を代入する。
		bag_name = slide_id+'.h5'
		# h5_file_pathに入力h5ファイルのパスを代入する。
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		# slide_file_pathにslideファイルのパスを代入する。slide_extの引数が.svsならsvsファイル、.ndpiなら.ndpiファイルを読み込もうとする。
		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
		# 全体の内、スライドの何番目を処理しているのか表示する。
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		# スライド名を表示する。
		print(slide_id)

		# no_auto_skipがFalseで、スライド名.ptが出力ptフォルダ内に入っている場合、
		# そのスライドはスキップするというメッセージを表示して処理を飛ばす。
		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		# output_pathに出力ファイルパスを代入する。
		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		
		# チェック用a
		print("h5_file_path:",h5_file_path)
		# チェック用a終わり

		# 時間測定開始
		time_start = time.time()
		# slideファイル(wsi)をopenslideで読み込む。
		wsi = openslide.open_slide(slide_file_path)
		# 出力ファイルパスに、上で定義しているcompute_w_loader関数の返り値を入れる。
		# 引数の中で、pretrainedが指定されていない。上の関数定義を見ると、デフォルトはpretrained=Trueなので
		# 基本的に学習済みの重みが使用されていると思われる。
		# この中で読まれるdataset_h5.pyのWhole_Slide_Bag_FPクラスの処理を見ると、
		# 引数target_patch_size=1以上を指定していたら、引数custom_downsampleを指定していても関係ないと思われる。
		output_file_path = compute_w_loader(h5_file_path, output_path, wsi, 
		model = model, batch_size = args.batch_size, verbose = 1, print_every = 20, 
		custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size)
		# 処理にかかった時間をtime_elapsedに入れる
		time_elapsed = time.time() - time_start
		# 出力ファイルパス(output_file_path)と、処理にかかった時間(time_elapsed)を表示する。
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
		file = h5py.File(output_file_path, "r")

		features = file['features'][:]
		# 保存したh5ファイルを読み込んで、特徴のサイズや座標のサイズを表示させる
		print('features size: ', features.shape)
		print('coordinates size: ', file['coords'].shape)
		# h5ファイルから読み込んだ内、特徴部分をnumpy配列からtorchテンソル形式に変換して、ptファイルとして保存する。
		features = torch.from_numpy(features)
		# bag_name=slide_id+.h5だったのを、.h5を取り除いてbag_baseに代入する。
		bag_base, _ = os.path.splitext(bag_name)
		# pt_filesフォルダに、bag_gase+.ptの名前で.ptファイルを保存する。
		torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))
		# .h5ファイルにはnumpy配列形式の特徴と座標が、.ptファイルにはテンソル形式の特徴が入っている。
		# 特徴の形状は基本的に[512,1024]、バッチの部分によって[471,1024]等になる。



