from __future__ import print_function

import numpy as np

import argparse

import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
from utils.eval_utils import initiate_model as initiate_model
from models.model_clam import CLAM_MB, CLAM_SB
from models.resnet_custom import resnet50_baseline
from types import SimpleNamespace
from collections import namedtuple
import h5py
import yaml
from wsi_core.batch_process_utils import initialize_df
from vis_utils.heatmap_utils import initialize_wsi, drawHeatmap, compute_from_patches
from wsi_core.wsi_utils import sample_rois
from utils.file_utils import save_hdf5

parser = argparse.ArgumentParser(description='Heatmap inference script')
parser.add_argument('--save_exp_code', type=str, default=None,
					help='experiment code')
parser.add_argument('--overlap', type=float, default=None)
parser.add_argument('--config_file', type=str, default="heatmap_config_template.yaml")
args = parser.parse_args()

def infer_single_slide(model, features, label, reverse_label_dict, k=1):
	features = features.to(device)
	with torch.no_grad():
		if isinstance(model, (CLAM_SB, CLAM_MB)):
			model_results_dict = model(features)
			logits, Y_prob, Y_hat, A, _ = model(features)
			Y_hat = Y_hat.item()

			if isinstance(model, (CLAM_MB,)):
				A = A[Y_hat]

			A = A.view(-1, 1).cpu().numpy()

		else:
			raise NotImplementedError

		print('Y_hat: {}, Y: {}, Y_prob: {}'.format(reverse_label_dict[Y_hat], label, ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()]))	
		# torch.topkは今回インストールしたsvm.topkとは別。
		probs, ids = torch.topk(Y_prob, k)
		probs = probs[-1].cpu().numpy()
		ids = ids[-1].cpu().numpy()
		preds_str = np.array([reverse_label_dict[idx] for idx in ids])

	return ids, preds_str, probs, A

def load_params(df_entry, params):
	for key in params.keys():
		if key in df_entry.index:
			dtype = type(params[key])
			val = df_entry[key] 
			val = dtype(val)
			if isinstance(val, str):
				if len(val) > 0:
					params[key] = val
			elif not np.isnan(val):
				params[key] = val
			else:
				pdb.set_trace()

	return params

def parse_config_dict(args, config_dict):
	if args.save_exp_code is not None:
		config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
	if args.overlap is not None:
		config_dict['patching_arguments']['overlap'] = args.overlap
	return config_dict

if __name__ == '__main__':
	# ここからスタート
	# コマンドで絶対パス指定しているため、args.config_fileには絶対パスが入る。
	# そのため、恐らくここは絶対パスにする必要はないはず。
	config_path = os.path.join('heatmaps/configs', args.config_file)
	#チェック用b2
	print("#config_path:",config_path)
	#チェック用b2終わり
	# config_template_win64.yamlなどのコマンド引数で指定したyamlファイルを読み込んでいる。
	# このyamlファイルには、プログラムで使用する各種引数が辞書形式で保存されている。
	# config_dictとしてconfig_file(yaml形式)を読み込む
	config_dict = yaml.safe_load(open(config_path, 'r'))
	# 上記parse_config_dict関数を呼び出す。save_exp_codeがあればconfig_dict内のexp_argumentsのsave_exp_codeに、引数のsave_exp_codeを入れる。
	# overlapがあれば、config_dict内のpatching_arguments内のoverlapに、引数のoverlapを入れる。
	config_dict = parse_config_dict(args, config_dict)

	# yamlファイル内の辞書のキーと値をひとつずつ取り出す。ただし、辞書のキーには値に更に辞書が
	# 辞書のキーには分類が書かれていて、値には更に複数引数の辞書が入っているという構造なので注意
	for key, value in config_dict.items():
		# もしもvalueの形式がdictの場合、つまり入れ子の辞書形式の場合は、以下を実施
		if isinstance(value, dict):
			# config_template内のキー、つまり分類名を表示する
			print('\n'+key)
			# 辞書形式の値のキーと値拾って繰り返す。
			for value_key, value_value in value.items():
				# 項目内のキーと値を表示する。
				print (value_key + " : " + str(value_value))
		else:
			# 再外部の辞書の値が、更に辞書形式でなければ、再外部のキーと値を表示する。
			print ('\n'+key + " : " + str(value))
			
	# 処理を続けるかどうかの確認を表示。yやyes等なら継続、n等なら終了、それ以外なら実装エラーとする。
	decision = input('Continue? Y/N ')
	if decision in ['Y', 'y', 'Yes', 'yes']:
		pass
	elif decision in ['N', 'n', 'No', 'NO']:
		exit()
	else:
		raise NotImplementedError

	args = config_dict
	patch_args = argparse.Namespace(**args['patching_arguments'])
	data_args = argparse.Namespace(**args['data_arguments'])
	model_args = args['model_arguments']
	# n_classesをyamlファイルに指定した値(「2」等)で更新
	model_args.update({'n_classes': args['exp_arguments']['n_classes']})
	model_args = argparse.Namespace(**model_args)
	exp_args = argparse.Namespace(**args['exp_arguments'])
	heatmap_args = argparse.Namespace(**args['heatmap_arguments'])
	sample_args = argparse.Namespace(**args['sample_arguments'])

	# ([256,256])等のタプルができる
	patch_size = tuple([patch_args.patch_size for i in range(2)])
	# patch_sizeに重ならない部分の割合(1-overlap)を掛けるとstep_size(patchsize[0]*(1-overlap),patchsize[1]*(1-overlap))になる
	# つまり、overlap=0の場合はpatch_size毎に重ならずにstepしていく。どう作用するか要確認
	step_size = tuple((np.array(patch_size) * (1-patch_args.overlap)).astype(int))
	print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(patch_size[0], patch_size[1], patch_args.overlap, step_size[0], step_size[1]))
	
	# preset引数に書かれたcsvファイルを読み込む(seg_level,use_otsu等の記載あり)
	preset = data_args.preset
	#チェック用b
	print("#data_args: ",data_args)
	print("#preset: ",preset)
	#チェック用b終わり
	def_seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False, 
					  'keep_ids': 'none', 'exclude_ids':'none'}
	def_filter_params = {'a_t':50.0, 'a_h': 8.0, 'max_n_holes':10}
	def_vis_params = {'vis_level': -1, 'line_thickness': 250}
	def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

	# presetファイルがあるときは、その記載に沿って以下の辞書を更新していく
	if preset is not None:
		# presetファイルをデータフレームとして読み込む(pd.read_csv)
		preset_df = pd.read_csv(preset)
		# def_seg_params辞書の更新
		for key in def_seg_params.keys():
			def_seg_params[key] = preset_df.loc[0, key]

		# def_filter_params辞書の更新
		for key in def_filter_params.keys():
			def_filter_params[key] = preset_df.loc[0, key]

		# def_vis_params辞書の更新(vis_level, line_thickness)
		for key in def_vis_params.keys():
			def_vis_params[key] = preset_df.loc[0, key]

		# def_patch_params辞書の更新
		for key in def_patch_params.keys():
			def_patch_params[key] = preset_df.loc[0, key]

	# テスト print
	print("data_args.data_dir:",data_args.data_dir)

	# slide_idが入ったprocess_listのファイル指定がない場合、
	# data_dirがlist型だった場合は、wsiが入った名称をリストとして取得
	if data_args.process_list is None:
		if isinstance(data_args.data_dir, list):
			slides = []
			for data_dir in data_args.data_dir:
				# slidesに、data_dir(今回は"input_wsi")内のファイル名のリスト(os.listdirで読み込むとリストになる)を結合する。
				slides.extend(os.listdir(data_dir))
		else:
			# list型でなければ、slidesにリスト形式かつ昇順でdata_dir内のwsiファイル名を入れる([Normal.ndpi,Tumor.ndpi等])
			# この場合、この時点では各要素に拡張子が残る。
			slides = sorted(os.listdir(data_args.data_dir))
		# インプットファイルとして、指定した拡張子のファイルのみをslidesリストの中に残す。
		slides = [slide for slide in slides if data_args.slide_ext in slide]
		# wsi_core\batch_process_utils.pyのinitialize_df関数を呼び出す。
		# batch_process_utils.pyのinitialize_df関数を呼び出し、スライドの数だけnumpy配列に実行のパラメータが入ったデータフレームがdfに入る。
		# initialize_dfで処理されるのは、presetファイルで指定した項目で、yamlファイルから読み込んだものは含まないと思われる。
		df = initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)
		
	else:
		# process_listにslide_idが入ったcsvのパスが入っている場合は、csvをデータフレーム形式で読み込み、batch_process_utils.pyのinitialize_df関数を呼び出し。
#               windows用に、以下のパスを変更
#		df = pd.read_csv(os.path.join('heatmaps/process_lists', data_args.process_list))
		df = pd.read_csv(os.path.join("C:\\Users\\akihi\\CLAM-master\\heatmaps\\process_lists", data_args.process_list))
		# wsi_core\batch_process_utils.pyのinitialize_df関数を呼び出す
		# 引数は、slideのリストが入ったファイル(df)と切り分け用のパラメータ群
		# initialize_dfで処理されるのは、presetファイルで指定した項目で、yamlファイルから読み込んだものは含まないと思われる。
		df = initialize_df(df, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)

	# maskにはslideの枚数と同じ数のTrueが入る。
	mask = df['process'] == 1
	# dfに新たな列としてmaskを上から順に並べたと考えて、
	# 上から順にmaskがTrueの行だけをdf内に残したものをprocess_stackデータフレームとする。
	# 全てTrueなら、dfは元々process_listファイルだが、initialize_df関数でpreset内の項目が追加されているので、
	# ➡process_stackも、process_listにしてしてある列＋presetで設定されている項目+process列という事になる。
	process_stack = df[mask].reset_index(drop=True)
	# totalには処理するスライド数が入る
	total = len(process_stack)
	print('\nlist of slides to process: ')
	print(process_stack.head(len(process_stack)))

	print('\ninitializing model from checkpoint')
	# モデルファイル(.pt)のパスが表示される
	ckpt_path = model_args.ckpt_path
	print('\nckpt path: {}'.format(ckpt_path))
	
	# initiate_fn引数がinitiate_modelの場合、eval_utils.pyのinitiate_model関数でモデルの初期化を行う
	if model_args.initiate_fn == 'initiate_model':
		model =  initiate_model(model_args, ckpt_path)
	else:
		raise NotImplementedError


	# 特徴抽出器としてfeature_extractorにresnet_custom.pyのresnet50_baseline関数(事前学習済)の返り値としてモデルが入る。
	# extract_features_fpでも同じ関数を呼び出すが、その時は「feature_extractor」ではなく「model」という名称にしている。
	feature_extractor = resnet50_baseline(pretrained=True)

	# evalモードなので、dropoutやbattch normがonになっている？
	# 参考：https://qiita.com/tatsuya11bbs/items/86141fe3ca35bdae7338
	feature_extractor.eval()
	device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print('Done!')

	# yamlファイルを元にしたdata_args.label_dictをlabel_dictに入れる。(例＝"tumor_tissue":1,"normal_tissue":0等)
	label_dict =  data_args.label_dict
	# class_labelsにlabel_dictのキーをリストにしていれる。(例＝["tumor_tissue","normal_tissue"])
	class_labels = list(label_dict.keys())
	# class_encodingsにlabel_dictの値をリストにしていれる。(例＝[1,0])
	class_encodings = list(label_dict.values())
	# reverse_label_dictをlabel_dictの値とキーを逆にした辞書にする。（例＝{1:"tumor_tissue",0:"normal_tissue"}）
	reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))} 

	# feature_extractorをcuda等に繋げる
	if torch.cuda.device_count() > 1:
		device_ids = list(range(torch.cuda.device_count()))
		feature_extractor = nn.DataParallel(feature_extractor, device_ids=device_ids).to('cuda:0')
	else:
		feature_extractor = feature_extractor.to(device)

	# 引数で指定している最終フォルダを作成する(exist_ok=Trueで既にフォルダが存在していてもエラーにならない。)
	os.makedirs(exp_args.production_save_dir, exist_ok=True)
	# 引数で指定しているrawフォルダを作成する(exist_ok=Trueで既にフォルダが存在していてもエラーにならない。)
	os.makedirs(exp_args.raw_save_dir, exist_ok=True)
	# ブロック状wsiの複数引数を表すblocky_wsi_kwargsに以下の様な辞書形式を入れる。
	# top_left:なし、bot_right:なし、patch_size:パッチサイズ(256,256)等、step_size:パッチサイズと同じ
	# custom_downsample:yamlのcustom_downsample,patch_level:yamlのpatch_level、use_center_shift:yamlのuse_center_shift(default=True)
	blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': patch_size, 'step_size': patch_size, 
	'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}

	# スライド枚数だけ繰り返し
	for i in range(len(process_stack)):
		# slide_nameにslide_idを入れる。つまりinput_wsiからの場合、Normal.ndpiやTumor.ndpiが入る。process_listファイルからの場合NormalやTumor。
		slide_name = process_stack.loc[i, 'slide_id']
		# slide_nameにyamlで設定している拡張子が入ってなかったら、拡張子を付ける(例＝Normal.ndpi)
		if data_args.slide_ext not in slide_name:
			slide_name+=data_args.slide_ext
		print('\nprocessing: ', slide_name)	

		# labelにはprocess_listファイルから読み込んだ場合、label列があるなら入っているはずなので各label(Tumor_tissue等)を入れる。
		# label列がない、process_listでない場合等はUnspecifiedを入れる。(use_heatmap_args=Trueなら-1が入るが、ここがTrueになることはそもそもないはず。)
		try:
			label = process_stack.loc[i, 'label']
		except KeyError:
			label = 'Unspecified'
		# slide_idから.ndpiを消す。ここで、ファイルから読み込んだ場合も、wsiの入ったフォルダから読み込んだ場合もslide_idから拡張子が消える。
		slide_id = slide_name.replace(data_args.slide_ext, '')

		# labelが文字列でない場合はgroupingにyamlのlabel_dictに指定したキーを入れる。(1ならtumor_tissue、0ならnormal_tissue)
		# labelが文字列なら、そのままlabelの文字列をgroupingに入れる。
		if not isinstance(label, str):
			grouping = reverse_label_dict[label]
		else:
			grouping = label

		# 最終出力先のディレクトリとして、groupingに入った名前のパスを作る。(.../tumor_tissueや.../normal_tissue)
		# windows例＝C:\Users\akihi\CLAM-master\heatmaps\heatmap_production_results\HEATMAP_OUTPUT\normal_tissue
		p_slide_save_dir = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, str(grouping))
		os.makedirs(p_slide_save_dir, exist_ok=True)

		# 生データ出力先のディレクトリとして、groupingに入った名前のパスを作る。(.../tumor_tissueや.../normal_tissue)
		r_slide_save_dir = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, str(grouping),  slide_id)
		os.makedirs(r_slide_save_dir, exist_ok=True)

		# ヒートマップ引数のuse_roiがTrueの場合、
		# x1,x2,y1,y2にprocess_stack内に指定の値を代入して、
		# top_left(左上),bot_right(右下)座標として入れる。
		if heatmap_args.use_roi:
			x1, x2 = process_stack.loc[i, 'x1'], process_stack.loc[i, 'x2']
			y1, y2 = process_stack.loc[i, 'y1'], process_stack.loc[i, 'y2']
			top_left = (int(x1), int(y1))
			bot_right = (int(x2), int(y2))
		# ヒートマップ引数のuse_roiがFalseの場合、
		# top_left=None bot_right=Noneとする。
		else:
			top_left = None
			bot_right = None
		
		print('slide id: ', slide_id)
		print('top left: ', top_left, ' bot right: ', bot_right)

		# 入力wsiフォルダパスがstringの場合は、スライド名を後に繋げてslide_pathとする。
		if isinstance(data_args.data_dir, str):
			slide_path = os.path.join(data_args.data_dir, slide_name)
		# 入力wsiフォルダパスが辞書形式の場合は、data_dir_keyにyamlファイル内の引数data_dir_keyの値を入れる。
		# slide_pathには、yamlの引数data_dir辞書の[data_dir_key]をキーとした値に、slide_nameを繋げたものを入れる。
		elif isinstance(data_args.data_dir, dict):
			data_dir_key = process_stack.loc[i, data_args.data_dir_key]
			slide_path = os.path.join(data_args.data_dir[data_dir_key], slide_name)
		else:
			raise NotImplementedError

		# mask_fileには、生データ出力フォルダ＋slide_id+_mask.pklを入れる。(拡張子(.ndpi等)が除去されているので、slide_id="Normal"等)
		mask_file = os.path.join(r_slide_save_dir, slide_id+'_mask.pkl')
		
		# Load segmentation and filter parameters
		# seg_paramsにpresetファイルベースのdef_seg_paramsのコピーを代入
		# filter_paramsやvis_paramsも同様
		seg_params = def_seg_params.copy()
		filter_params = def_filter_params.copy()
		vis_params = def_vis_params.copy()

		# 本ファイル上部のload_params関数で、
		# 必要に応じて上記のseg_params, filter_params, vis_paramsを更新する。
		# それぞれ、process_stackの中にpresetファイルと同じ項目の値が指定されていたら、process_stack内の値で更新する。
		# process_stack内の値で更新する。
		seg_params = load_params(process_stack.loc[i], seg_params)
		filter_params = load_params(process_stack.loc[i], filter_params)
		# vis_paramsのキーと同じものがprocess_stackの行に含まれていなければ変わらない。
		# process_stackには全て1のprocess列以外に、presetファイルで指定した各パラメータが含まれるので、
		# vis_paramsもpresetファイルで指定した形に変更される。
		vis_params = load_params(process_stack.loc[i], vis_params)

		# keep_idsもpresetファイルに記載の値を文字列にしたものに変更される。(基本はnoneのはず)
		keep_ids = str(seg_params['keep_ids'])
		# 文字列の長さが0より長く、noneでもない場合、カンマ区切りのnumpy配列にする。
		if len(keep_ids) > 0 and keep_ids != 'none':
			seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
		# keep_idsの長さ0かnoneの場合、空にする。
		else:
			seg_params['keep_ids'] = []

		# exclude_idsもpresetファイルに記載の値を文字列にしたものに変更される。(基本はnoneのはず)
		exclude_ids = str(seg_params['exclude_ids'])
		# 文字列の長さが0より長く、noneでもない場合、カンマ区切りのnumpy配列にする。
		if len(exclude_ids) > 0 and exclude_ids != 'none':
			seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
		# keep_idsの長さ0かnoneの場合、空にする。
		else:
			seg_params['exclude_ids'] = []

		# seg_paramsが表示される。
		# ここのseg_paramsはpresetファイルの値が元となっている。
		for key, val in seg_params.items():
			print('{}: {}'.format(key, val))

		# filter_paramsが表示される。
		# ここのfilter_paramsはpresetファイルの値が元となっている。
		for key, val in filter_params.items():
			print('{}: {}'.format(key, val))

		# vis_paramsのvis_levelとline_thicknessが表示される。
		# ここのvis_paramsはpresetファイルの値が元となっている。
		for key, val in vis_params.items():
			print('{}: {}'.format(key, val))
		
		print('Initializing WSI object')
		# wsi_objectはheatmap_utils.pyのinitialize_wsi関数の返り値
		# 関数の内部ではWholeSlideImage.pyのWholeSlideImageクラスが呼ばれている。
		wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
		print('Done!')

		wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]

		# the actual patch size for heatmap visualization should be the patch size * downsample factor * custom downsample factor
		vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_ref_downsample) * patch_args.custom_downsample).astype(int))

		block_map_save_path = os.path.join(r_slide_save_dir, '{}_blockmap.h5'.format(slide_id))
		mask_path = os.path.join(r_slide_save_dir, '{}_mask.jpg'.format(slide_id))
		if vis_params['vis_level'] < 0:
			# wsi_object.wsi=スライドのファイルをopenslide.open_slideで開いたイメージ
			best_level = wsi_object.wsi.get_best_level_for_downsample(32)
			vis_params['vis_level'] = best_level
			# vis_params['vis_level']はpresetファイルの値がベース(-1 -> 5)
			# vis_params['vis_level']を確認-> 「vis_level：5」を確認済
			print("check..vis_params['vis_level']:",vis_params['vis_level'])
		# WholeSlideImage.pyのinitialize_wsiクラスのvisWSIメソッドを呼び出す。
		mask = wsi_object.visWSI(**vis_params, number_contours=True)
		# presetのvis_levelが-1の場合の-1 -> 5は、スライドの穴や細胞境界を切り分けた画像の保存に用いられる。
		# 例：C:\Users\akihi\CLAM-master\heatmaps\heatmap_raw_results\HEATMAP_OUTPUT\Unspecified\Tumor\Tumor_mask.jpg
		mask.save(mask_path)
		
		features_path = os.path.join(r_slide_save_dir, slide_id+'.pt')
		h5_path = os.path.join(r_slide_save_dir, slide_id+'.h5')
	

		##### check if h5_features_file exists ######
		if not os.path.isfile(h5_path) :
			_, _, wsi_object = compute_from_patches(wsi_object=wsi_object, 
											model=model, 
											feature_extractor=feature_extractor, 
											batch_size=exp_args.batch_size, **blocky_wsi_kwargs, 
											attn_save_path=None, feat_save_path=h5_path, 
											ref_scores=None)				
		
		##### check if pt_features_file exists ######
		if not os.path.isfile(features_path):
			file = h5py.File(h5_path, "r")
			features = torch.tensor(file['features'][:])
			torch.save(features, features_path)
			file.close()

		# load features 
		features = torch.load(features_path)
		process_stack.loc[i, 'bag_size'] = len(features)
		
		wsi_object.saveSegmentation(mask_file)
		Y_hats, Y_hats_str, Y_probs, A = infer_single_slide(model, features, label, reverse_label_dict, exp_args.n_classes)
		del features
		
		if not os.path.isfile(block_map_save_path): 
			file = h5py.File(h5_path, "r")
			coords = file['coords'][:]
			file.close()
			asset_dict = {'attention_scores': A, 'coords': coords}
			block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')
		
		# save top 3 predictions
		for c in range(exp_args.n_classes):
			process_stack.loc[i, 'Pred_{}'.format(c)] = Y_hats_str[c]
			process_stack.loc[i, 'p_{}'.format(c)] = Y_probs[c]

		# 出力フォルダとファイルの一部がlinux形式で指定されており、相対パス
		# 絶対パスに変更しておく,process_listは.yamlファイル内で絶対パスで書かれているため、今回出力先を求める際はファイル名のみ(os.path.basename()で変換)にしておく。
		os.makedirs('C:\\Users\\akihi\\CLAM-master\\heatmaps\\results\\', exist_ok=True)
#		os.makedirs('heatmaps/results/', exist_ok=True)
		if data_args.process_list is not None:
			process_stack.to_csv('C:\\Users\\akihi\\CLAM-master\\heatmaps\\results\\{}.csv'.format(os.path.basename(data_args.process_list).replace('.csv', '')), index=False)
#			process_stack.to_csv('heatmaps/results/{}.csv'.format(data_args.process_list.replace('.csv', '')), index=False)
		else:
			process_stack.to_csv('C:\\Users\\akihi\\CLAM-master\\heatmaps\\results\\{}.csv'.format(exp_args.save_exp_code), index=False)
#			process_stack.to_csv('heatmaps/results/{}.csv'.format(exp_args.save_exp_code), index=False)
		
		file = h5py.File(block_map_save_path, 'r')
		dset = file['attention_scores']
		coord_dset = file['coords']
		scores = dset[:]
		coords = coord_dset[:]
		file.close()

		samples = sample_args.samples
		for sample in samples:
			if sample['sample']:
				tag = "label_{}_pred_{}".format(label, Y_hats[0])
				sample_save_dir =  os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, 'sampled_patches', str(tag), sample['name'])
				os.makedirs(sample_save_dir, exist_ok=True)
				print('sampling {}'.format(sample['name']))
				sample_results = sample_rois(scores, coords, k=sample['k'], mode=sample['mode'], seed=sample['seed'], 
					score_start=sample.get('score_start', 0), score_end=sample.get('score_end', 1))
				for idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
					print('coord: {} score: {:.3f}'.format(s_coord, s_score))
					patch = wsi_object.wsi.read_region(tuple(s_coord), patch_args.patch_level, (patch_args.patch_size, patch_args.patch_size)).convert('RGB')
					patch.save(os.path.join(sample_save_dir, '{}_{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, slide_id, s_coord[0], s_coord[1], s_score)))

		wsi_kwargs = {'top_left': top_left, 'bot_right': bot_right, 'patch_size': patch_size, 'step_size': step_size, 
		'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}

		heatmap_save_name = '{}_blockmap.tiff'.format(slide_id)
		if os.path.isfile(os.path.join(r_slide_save_dir, heatmap_save_name)):
			pass
		else:
			heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object, cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, use_holes=True, binarize=False, vis_level=-1, blank_canvas=False,
							thresh=-1, patch_size = vis_patch_size, convert_to_percentiles=True)
		
			heatmap.save(os.path.join(r_slide_save_dir, '{}_blockmap.png'.format(slide_id)))
			del heatmap

		save_path = os.path.join(r_slide_save_dir, '{}_{}_roi_{}.h5'.format(slide_id, patch_args.overlap, heatmap_args.use_roi))

		if heatmap_args.use_ref_scores:
			ref_scores = scores
		else:
			ref_scores = None
		
		if heatmap_args.calc_heatmap:
			compute_from_patches(wsi_object=wsi_object, clam_pred=Y_hats[0], model=model, feature_extractor=feature_extractor, batch_size=exp_args.batch_size, **wsi_kwargs, 
								attn_save_path=save_path,  ref_scores=ref_scores)

		if not os.path.isfile(save_path):
			print('heatmap {} not found'.format(save_path))
			# デフォルトはuse_roiはFalse
			if heatmap_args.use_roi:
				save_path_full = os.path.join(r_slide_save_dir, '{}_{}_roi_False.h5'.format(slide_id, patch_args.overlap))
				print('found heatmap for whole slide')
				save_path = save_path_full
			else:
				continue

		file = h5py.File(save_path, 'r')
		dset = file['attention_scores']
		coord_dset = file['coords']
		# scoresには、h5ファイル内のキー['attention_scores']の値をリストとして保管
		scores = dset[:]
		# coordsには、h5ファイル内のキー['coords']の値をリストとして保管。これはパッチの座標が入っていると思われる。
		coords = coord_dset[:]
		file.close()

		heatmap_vis_args = {'convert_to_percentiles': True, 'vis_level': heatmap_args.vis_level, 'blur': heatmap_args.blur, 'custom_downsample': heatmap_args.custom_downsample}
		if heatmap_args.use_ref_scores:
			heatmap_vis_args['convert_to_percentiles'] = False

		heatmap_save_name = '{}_{}_roi_{}_blur_{}_rs_{}_bc_{}_a_{}_l_{}_bi_{}_{}.{}'.format(slide_id, float(patch_args.overlap), int(heatmap_args.use_roi),
																						int(heatmap_args.blur), 
																						int(heatmap_args.use_ref_scores), int(heatmap_args.blank_canvas), 
																						float(heatmap_args.alpha), int(heatmap_args.vis_level), 
																						int(heatmap_args.binarize), float(heatmap_args.binary_thresh), heatmap_args.save_ext)


		if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
			pass
		
		else:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
			# heatmap_utils.pyのdrawHeatmap関数を呼び出す
			heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object,  
						          cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, **heatmap_vis_args, 
						          binarize=heatmap_args.binarize, 
						  		  blank_canvas=heatmap_args.blank_canvas,
						  		  thresh=heatmap_args.binary_thresh,  patch_size = vis_patch_size,
						  		  overlap=patch_args.overlap, 
						  		  top_left=top_left, bot_right = bot_right)
			if heatmap_args.save_ext == 'jpg':
				heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
			else:
				heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))
		
		# もしもyamlファイル内のheatmap_argumentsのsave_orig引数がTrueなら実行
		if heatmap_args.save_orig:
			# yamlファイル内のheatmap_argumentsのvis_level引数が0以上ならそのまま
			if heatmap_args.vis_level >= 0:
				vis_level = heatmap_args.vis_level
			# yamlファイル内のheatmap_argumentsのvis_level引数が-1以下なら
			# presetファイル内のvis_levelが用いられる。ただし、preset内のvis_levelが-1の場合は5になる。
			else:
				vis_level = vis_params['vis_level']
			# heatmap_save_nameをTumor_orig_1.jpg等にする。次にオリジナルの細胞画像をvis_levelで保存すると思われる。
			heatmap_save_name = '{}_orig_{}.{}'.format(slide_id,int(vis_level), heatmap_args.save_ext)
			if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
				pass
			else:
				# WholeSlideImage.pyのinitialize_wsiクラスのvisWSIメソッドを呼び出す。
				heatmap = wsi_object.visWSI(vis_level=vis_level, view_slide_only=True, custom_downsample=heatmap_args.custom_downsample)
				if heatmap_args.save_ext == 'jpg':
					heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
				else:
					heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))

	with open(os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, 'config.yaml'), 'w') as outfile:
		yaml.dump(config_dict, outfile, default_flow_style=False)


