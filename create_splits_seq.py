import pdb
import os
import pandas as pd
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping'])
# 検証用ラベルの割合
parser.add_argument('--val_frac', type=float, default= 0.1,
                    help='fraction of labels for validation (default: 0.1)')
# テスト用のラベルの割合
parser.add_argument('--test_frac', type=float, default= 0.1,
                    help='fraction of labels for test (default: 0.1)')

args = parser.parse_args()

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
#    windowsなので、ファイルパスを/から\\に変更する。また、絶対パスにしておく
#    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'C:\\Users\\akihi\\CLAM-master\\dataset_csv\\tumor_vs_normal_dummy_clean.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=True,
                            ignore=[])
    # チェック用f
    print("dataset:",dataset)
    # チェック用f終わり

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
#    windowsなので、ファイルパスを/から\\に変更する。また、絶対パスにしておく
#    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'C:\\Users\\akihi\\CLAM-master\\dataset_csv\\tumor_subtyping_dummy_clean.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

else:
    raise NotImplementedError

# num_slides_clsはlabelに[1]だけの状態からnp.where等で繰り返し[0]と[1]を検索なので空と[0]になっている。
# そのため、len(cls_ids)は0(空の分)と1([0]の分)つまり、len(cls_ids)=[0 1]となりnum_slides_clsは[0 1]のnumpy配列となる。
num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
# val_numはnum_slides_cls([0 1])×args.val_frac(0.1) = [0 0.1]だが、np.roundにより[0 0]になっている。
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
test_num = np.round(num_slides_cls * args.test_frac).astype(int)

# チェック用j
print("#patient_cls_ids:",dataset.patient_cls_ids)
print("#num_slides_cls:",num_slides_cls)
print("#val_num:",val_num)
print("#test_num:",test_num)
# チェック用j終わり


if __name__ == '__main__':
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]

    # チェック用k
    print("#label_fracs:",label_fracs)
    # チェック用k終わり
    
    for lf in label_fracs:
        # チェック用l
        print("#lf:",lf)
        # チェック用l終わり

        # windows用にパスを/から\\に置き換える。
        split_dir = 'splits\\'+ str(args.task) + '_{}'.format(int(lf * 100))
#        split_dir = 'splits/'+ str(args.task) + '_{}'.format(int(lf * 100))
        # チェック用d
        print("#split_dir:",split_dir)
        # チェック用d終わり
        # 念のため絶対パスに指定する(分割フォールド用のフォルダを保存)
        os.makedirs(os.path.join("C:\\Users\\akihi\\CLAM-master",split_dir), exist_ok=True)
#        os.makedirs(split_dir, exist_ok=True)        
        dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf)
        for i in range(args.k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            # チェック用g
            print("#descriptor_df:",descriptor_df)
            # チェック用g終わり
            splits = dataset.return_splits(from_id=True)
            # チェック用e
            print("#splits:",splits)
            # チェック用e終わり
            # 絶対パスにしておく(分割フォールドを指定するcsvファイル(splits_0.csv等)を,上記os.mkdirsで作成したフォルダに保存)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(os.path.join("C:\\Users\\akihi\\CLAM-master",split_dir), 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(os.path.join("C:\\Users\\akihi\\CLAM-master",split_dir), 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            descriptor_df.to_csv(os.path.join(os.path.join("C:\\Users\\akihi\\CLAM-master",split_dir), 'splits_{}_descriptor.csv'.format(i)))
#            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
#            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
#            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))



