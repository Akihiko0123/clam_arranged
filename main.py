from __future__ import print_function

import argparse
import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np


def main(args):
    # 無ければ引数でディレクトリを作成する。
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    # 引数k_startが-1の場合はstartが0になる。
    if args.k_start == -1:
        start = 0
    # 引数k_startが-1以外の場合はstartが指定した値になる
    else:
        start = args.k_start
    # 引数k_endが-1の時は,endが引数kと同じ値になる。(kはフォールド数関係)
    if args.k_end == -1:
        end = args.k
    # 引数k_endが-1以外の場合はendがk_endになる。
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    # startからendまでがフォールドの番号になる。(start=5,end=10の場合、array([5,6,7,8,9]))
    folds = np.arange(start, end)
    # foldの番号順に繰り返す
    for i in folds:
        # seed_torchは下の方でに定義している関数。引数のseed値で関連すると思われるすべての乱数を固定している。
        seed_torch(args.seed)
        # dataset_generic.pyのreturn_splitsからtrain,val,testのdatasetを取得している。
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                # windows用のパス名＋絶対パス変更
                # foldの分割の仕方が書かれたcsvファイルを、各foldで読み込む。
                csv_path='{}\\splits_{}.csv'.format(args.split_dir, i))
#                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))

        # datasetsに上記で読み込んだtrain_dataset等をタプルにしている。        
        datasets = (train_dataset, val_dataset, test_dataset)
        # チェック用v train_datasetの長さが0しかない。原因調査する
        print("#datasets:",datasets)
        print("#len_datasets:",len(datasets))
        print("#train_dataset:",train_dataset)
        print("#len_train_dataset:",len(train_dataset))
        # チェック用v終わり
        # utils.core_utils.pyのtrain関数を呼び出している。引数は各データセットのタプル(datasets)と、fold番号(i)、プログラム実行引数(args)
        results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
# 出力フォルダのデフォルトパスをwindowsの絶対パスにしておく。(linux形式でもパス記号がきちんと認識されている様子)
parser.add_argument('--results_dir', default='C:\\Users\\akihi\\CLAM-master\\results', help='results directory (default: C:\\Users\\akihi\\CLAM-master\\results)')
#parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=False, help='enabel dropout (p=0.25)')
# bag_lossがsvmならtop1 smoothSVM(上位1位が正解の確率)でロスを計算する。ceならクロスエントロピー誤差でロス計算。
# デフォルトはクロスエントロピー誤差。(ただ特にceかどうかでは判断していないような気もする[core_utils.py train関数参照])
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
# モデルタイプを、clam_sb(シングルブランチ),clam_mb(マルチブランチ),milから選ぶ、デフォルトはclam_sb
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
# モデルサイズをsmallかbigで指定、デフォルトはsmall
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping'])
### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
# instanceレベルのクラスタリングの損失関数(svmかceかNone、デフォルトはNone)
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, 
                     help='subtyping problem')
# bag_weightはbagレベルの損失についてのclamの重み係数。デフォルトは0.7
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
# clamのサンプルとなるpositive/negativeパッチの数(要確認)、デフォルトは8
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# プログラムで使う全ての乱数を指定したseed値に統一する関数(完全再現には大切)
def seed_torch(seed=7):
    import random
    # ランダム値をseed=7を初期値として固定している。(引数指定していればその値)
    random.seed(seed)
    # 環境変数('PYTHONHASHSEED')をseed値に設定している。
    # ハッシュベースでpythonプログラムの実行結果の再現性を担保するために用いるとのこと。
    # ただ、osの環境変数なので、次回のプログラム実行時には影響するが、
    # pythonが実行される前に変更しなければ、今回実行時には影響しないのでは？とも思う。
    # 参考記事:https://qiita.com/RIRO/items/29e5d7ffe464c0a4a630
    #         :https://teratail.com/questions/291319
    os.environ['PYTHONHASHSEED'] = str(seed)
    # numpyの乱数シードを引数seedに固定、np.random.stateも疑似乱数を固定するインスタンスで最近はこちらの方が良いとされている様子。
    np.random.seed(seed)
    # pytorchを用いたネットワークの重みの初期値にseedを用いている。再現性を持たせるため。
    torch.manual_seed(seed)
    if device.type == 'cuda':
        # 今はtorch.manual_seed(seed)でCPU,GPU含む全てのRNGに対して乱数値を固定している様子。
        # 以前は使用中GPUの乱数発生にseedを設定していた。参考：https://pytorch.org/docs/stable/notes/randomness.html
        torch.cuda.manual_seed(seed)
        # 以前は複数のGPU全てに対しseedで乱数固定していた。
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # ベンチマークモード。Trueでオートチューナーがネットワークの構成に最適なアルゴリズムを見つけ高速化するとのこと。
    # CNNのようにネットワークの入力サイズが変化しない場合はTrueにするのが推奨とのこと。
    torch.backends.cudnn.benchmark = False
    # cudaディープニューラルネットワークのloss値が毎回同じになるように固定している。(決定論的振る舞いをする)
    torch.backends.cudnn.deterministic = True

# 乱数を統一するseed_torch関数を呼び出す。seed値を引数(args.seed)の値に固定する。
seed_torch(args.seed)

# encoding_size=1024に設定する
encoding_size = 1024
# settingsにnum_splits=分割数、k_start=スタートfold番号、k_end=終了fold番号…を辞書形式で代入する。
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

if args.model_type in ['clam_sb', 'clam_mb']:
   # model_typeがclam_sb又はclam_mbのどちらかであれば、bag_weightとinst_loss, Bを追加する。
   settings.update({'bag_weight': args.bag_weight,
                    'inst_loss': args.inst_loss,
                    'B': args.B})

print('\nLoad Dataset')

# task引数の値が'task_1_tumor_vs_normal'の場合、クラス数を2にする。(n_classes=2)
if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    # windowsのため、パス記号を/から\\に変更する。また、絶対パスにしておく。
    # dataset_generic.pyのGeneric_MIL_Datasetクラスのインスタンスがdataset
    dataset = Generic_MIL_Dataset(csv_path = 'C:\\Users\\akihi\\CLAM-master\\dataset_csv\\tumor_vs_normal_dummy_clean.csv',
#    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
#                           'tumor_vs_normal_resnet_features'フォルダ内にpt_filesがあるという指定なので注意
                            data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=False,
                            ignore=[])
# task引数の値が'task_2_tumor_subtyping'の場合、クラス数を3にする。(n_classes=3)
elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    # windowsのため、パス記号を/から\\に変更する。また、絶対パスにしておく。
    dataset = Generic_MIL_Dataset(csv_path = 'C:\\Users\\akihi\\CLAM-master\\dataset_csv\\tumor_subtyping_dummy_clean.csv',
#    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
#                           'tumor_subtyping_resnet_features'フォルダ内にpt_filesがあるという指定なので注意
                            data_dir= os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= False,
                            ignore=[])

    if args.model_type in ['clam_sb', 'clam_mb']:
        assert args.subtyping 

# task引数が上記のいずれでもないなら、実装エラーを出力する。        
else:
    raise NotImplementedError
    
# results_dirのデフォルトはresultフォルダだが、なければ作成する。
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

# 結果のディレクトリ名。最後が1の場合は用いたseedが1ということになる。
args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

# 引数で指定していない場合に、task引数やlabel_frac引数等から下記のフォルダにデータを分割するためのファイルを取りに行くパスを作る。
# 引数で指定している場合は、"splits"に指定したargs.split_dirを繋げてパスを作る。
if args.split_dir is None:
    #絶対パスに変更しておく
    args.split_dir = os.path.join('C:\\Users\\akihi\\CLAM-master\\splits', args.task+'_{}'.format(int(args.label_frac*100)))
    #args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    #絶対パスに変更しておく
    args.split_dir = os.path.join('C:\\Users\\akihi\\CLAM-master\\splits', args.split_dir)
#    args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})

# experiment_~.txtファイルにsetting(実行に用いたオプション)を保存する。
with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

# importでなく実行された場合に、main関数を呼び出して実行する。
if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")
# 思ったよりシンプルかもしれない。
# feature_extractedでresnetを通して出力された途中結果を読み込み、
# attention形状のシンプルなネットワークに通して、最終的にクラス数(2や3)で出力される。
# ただ、クラスタリングしていると思ったのに、どこでしているのか分からなかった。恐らくinst_loss等がクラスタリングに関係するはず
