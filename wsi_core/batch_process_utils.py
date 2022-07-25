import pandas as pd
import numpy as np
import pdb
	
'''
initiate a pandas df describing a list of slides to process
args:
	slides (df or array-like): 
		array-like structure containing list of slide ids, if df, these ids assumed to be
		stored under the 'slide_id' column
	seg_params (dict): segmentation paramters 
	filter_params (dict): filter parameters
	vis_params (dict): visualization paramters
	patch_params (dict): patching paramters
	use_heatmap_args (bool): whether to include heatmap arguments such as ROI coordinates
'''
# create_patches.py, create_heatmaps.py等から呼び出される。presetファイルに指定したパラメータについては、スライド毎にパラメータを設定する辞書を作る。
# 各キーにnumpy配列形式で[1,1,1]のようにスライドの数と同じ数のパラメータが入る。例：3枚の場合、'seg_level': array([-1, -1, -1], dtype=int8)}
# use_heatmap_argsは呼び出し元でもFalseであり、yamlファイルなどでも指定できる引数ではないので、create_heatmaps.pyにおいては基本的にFalseと思われる。
def initialize_df(slides, seg_params, filter_params, vis_params, patch_params, 
	use_heatmap_args=False, save_patches=False):

	# slidesの長さ、つまりスライド枚数
	total = len(slides)
	'''
	初期パッチ作成の場合
	'''
	# slidesがリスト形式のはずなので、ここは該当しない
	'''
	ヒートマップの場合
	'''
	# slidesがyaml中process_listに記載のcsvファイル(基本dataset_csvフォルダ内)
	# から読み込んだ形式(データフレーム型)なら、
	# スライドID＝slidesDF内のslide_id列の値(wsiファイル名)と同じ値がリストで入る。
	if isinstance(slides, pd.DataFrame):
		slide_ids = slides.slide_id.values
	# スライドがprocess_listがなく、スライドがdata_dirから作ったリストなら(データフレーム型でなければ)、
	# slide_idsにslides(wsiファイル名)リストを入れる。
	else:
		slide_ids = slides
	# default_df_dict{'process':[1, 1, 1]}のように、'process'にはスライドの枚数
	# の長さの全要素に1を代入したnumpy配列が入る。(total=スライド枚数)
	# つまりdefault_df_dict = {'slide_id':[Tumor,Normal], 'process': array([1,1],dtype=uint8)}
	default_df_dict = {'slide_id': slide_ids, 'process': np.full((total), 1, dtype=np.uint8)}

	# create_heatmaps.pyから呼び出された時点ではuse_heatmap_argsはFalseなので該当しないと思われる。(必要なら確認)
	# use_heatmap_args=Trueの場合、default_df_dict辞書に'label':array([-1,-1,-1,...(スライドの数)])を追加する。dtype=np.uint8やint8ではない
	# initiate empty labels in case not provided
	if use_heatmap_args:
		default_df_dict.update({'label': np.full((total), -1)})
	
	# 設定パラメータを各スライド分に展開する。
	# パラメータごとに、項目数がスライド枚数のnumpy配列の中に、各パラメータの値を入れていく。
	# 全体としてはdefault_df_dict辞書の中にキーと値(numpy配列)が入っていく。
	# 基本的にはpresetファイルの値に更新されていると思われる。
	default_df_dict.update({
		'status': np.full((total), 'tbp'),
		# seg params
		'seg_level': np.full((total), int(seg_params['seg_level']), dtype=np.int8),
		'sthresh': np.full((total), int(seg_params['sthresh']), dtype=np.uint8),
		'mthresh': np.full((total), int(seg_params['mthresh']), dtype=np.uint8),
		'close': np.full((total), int(seg_params['close']), dtype=np.uint32),
		'use_otsu': np.full((total), bool(seg_params['use_otsu']), dtype=bool),
		'keep_ids': np.full((total), seg_params['keep_ids']),
		'exclude_ids': np.full((total), seg_params['exclude_ids']),
		
		# filter params
		'a_t': np.full((total), int(filter_params['a_t']), dtype=np.float32),
		'a_h': np.full((total), int(filter_params['a_h']), dtype=np.float32),
		'max_n_holes': np.full((total), int(filter_params['max_n_holes']), dtype=np.uint32),

		# vis params
		'vis_level': np.full((total), int(vis_params['vis_level']), dtype=np.int8),
		'line_thickness': np.full((total), int(vis_params['line_thickness']), dtype=np.uint32),

		# patching params
		'use_padding': np.full((total), bool(patch_params['use_padding']), dtype=bool),
		'contour_fn': np.full((total), patch_params['contour_fn'])
		})

	# デフォルトではsave_patches=Falseのようなので、パッチは保存されないと思われる。
	if save_patches:
		default_df_dict.update({
			'white_thresh': np.full((total), int(patch_params['white_thresh']), dtype=np.uint8),
			'black_thresh': np.full((total), int(patch_params['black_thresh']), dtype=np.uint8)})

	# デフォルトではFalseと思われる。
	# use_heatmap_args=Trueの場合、default_df_dict辞書に'x1','x2','y1','y2'を追加する。それぞれ要素NANのnumpy配列
	if use_heatmap_args:
		# initiate empty x,y coordinates in case not provided
		default_df_dict.update({'x1': np.empty((total)).fill(np.NaN), 
			'x2': np.empty((total)).fill(np.NaN), 
			'y1': np.empty((total)).fill(np.NaN), 
			'y2': np.empty((total)).fill(np.NaN)})


	# slidesがprocess_list引数から読み込んだcsv(データフレーム形式)の場合、default_df_dictをデータフレームにして、一時コピーとする。
	# csv内に、上記パラメータに関する指定がある場合は以下に続く
	if isinstance(slides, pd.DataFrame):
		temp_copy = pd.DataFrame(default_df_dict) # temporary dataframe w/ default params
		# find key in provided df
		# if exist, fill empty fields w/ default values, else, insert the default values as a new column
		for key in default_df_dict.keys(): 
			if key in slides.columns:
				# csv内のキー＝default_df_dictのキーの項目について、
				# maskにcsvのキーの値が空かどうかのデータを作る。注) 空でなければFalse、空ならTrue
				'''
				例：
				0    False
                1    False
                Name: slide_id, dtype: bool
				'''
				mask = slides[key].isna()
				slides.loc[mask, key] = temp_copy.loc[mask, key]
			else:
				# default_df_dictのkeysに一致する値がslidesの列になければ、slidesの最後列に列名(キー)と値を追加する->slidesにパラメータが入る。
				# slidesに全パラメータが入る。例えばlabel列が既にあった場合はlabe列はそのまま残す。
				slides.insert(len(slides.columns), key, default_df_dict[key])
	else:
		# slidesがinput_wsiから持ってきたリストの場合、データフレーム形式のdefault_df_dictで上書きする。
		slides = pd.DataFrame(default_df_dict)
	# 全部のパラメータが入ったデータフレームをslidesに代入し、呼び出し元に返す	
	return slides