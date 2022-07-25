import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_SB, CLAM_MB
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# create_heatmaps.py等から呼び出される。
def initiate_model(args, ckpt_path):
    print('Init Model')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    
    # 使用するモデルの種類("clam_sb"等)からモデルのインスタンスを作る。(dropoutやクラス数、サイズを引数としておく)
    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})
    if args.model_type =='clam_sb':
        model = CLAM_SB(**model_dict)
    elif args.model_type =='clam_mb':
        model = CLAM_MB(**model_dict)
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    print_network(model)

#   GPUではなくcpuを使用するようにtorch.loadを修正
#    ckpt = torch.load(ckpt_path)
    # main.pyで作成したモデルを読み込み、ckptに入れる。
    ckpt = torch.load(ckpt_path,map_location=torch.device('cpu'))
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    # modelに保存した情報を読み込む
    # 大枠ではload_state_dict(torch.load(ckpt_path))のような形式と言える。この方式ならモデル保存時のディレクトリ構造やGPUの種類などに依存しない。
    model.load_state_dict(ckpt_clean, strict=True)

    model.relocate()
    model.eval()
    return model

def eval(dataset, args, ckpt_path):
    # 上で定義しているinitiate_model関数を用いている。
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    # utils.pyのget_simple_loader関数を呼び出している。
    loader = get_simple_loader(dataset)
    # 以下に定義されているsummary関数を呼び出している。
    patient_results, test_error, auc, df, _ = summary(model, loader, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    return model, patient_results, test_error, auc, df

def summary(model, loader, args):
    # core_utils.pyに定義されているAccuracy_Loggerクラスのインスタンスacc_loggerを作成している。
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    # loaderからスライド番号全体を取得している。
    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        # スライド番号全体から、本for文の繰り返し番号(batch_idx)に該当するスライド番号を取得している。
        slide_id = slide_ids.iloc[batch_idx]
        # 以下with torch.no_grad()のネスト内で定義した変数は、自動的にrequires_grad=Falseとなるとのこと。
        # eval時によく使われるようで、計算グラフを作らないようにしており、結果的に勾配計算されないようになる様子。
        # https://qiita.com/kaba/items/da5ff6d93e5147412613
        with torch.no_grad():
            # 恐らく、モデルのforward関数を実行したと考えて良い？
            logits, Y_prob, Y_hat, _, results_dict = model(data)
        
        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        # all_labelsという名のnumpy配列の繰り返し番号(batch_idx)の位置に、その番号のラベル値(1とか0とか)を入れる。
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        
        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else: 
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, df, acc_logger
