import pickle
import h5py

def save_pkl(filename, save_object):
	writer = open(filename,'wb')
	pickle.dump(save_object, writer)
	writer.close()

def load_pkl(filename):
	loader = open(filename,'rb')
	file = pickle.load(loader)
	loader.close()
	return file


def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    # extract_features_fp.pyから呼び出された場合、書き込みモード(w)で出力先ファイルを開く
    file = h5py.File(output_path, mode)
    # キーと値を順番に繰り返す(0."features":features,1."coords":coords)
    for key, val in asset_dict.items():
        data_shape = val.shape
        # キー"features"や"coords"等がファイルにない場合は、create_datasetでキーや値の中身、形状等を保存する。
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path