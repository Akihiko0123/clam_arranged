import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np

class MIL_fc(nn.Module):
    # n_classesの初期値は2,このMIL_fcが呼ばれた時点でn_classesの初期値は2以下、
    # 2を超えていたらMIL_fc_mcが呼ばれる。
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes = 2, top_k=1):
        super(MIL_fc, self).__init__()
        # n_classes＝2でないとエラー終了する。
        assert n_classes == 2
        # size_dictが"small"なら[1024,512]にする。"big"の場合は？
        self.size_dict = {"small": [1024, 512]}
        # sizeを引数model_size(->size_arg)で指定した値にする。
        size = self.size_dict[size_arg]
        # fcを全結合層と活性化関数Reluのリストにする。
        # DropoutがTrueならfcに25%のDropout層を追加する。
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))

        # fcに恐らく出力層となる全結合層を追加する。
        fc.append(nn.Linear(size[1], n_classes))
        # 上記fcリストからself.classifierというモデルを作成している。
        self.classifier= nn.Sequential(*fc)
        # utils.pyの関数で重みを初期化している。
        initialize_weights(self)
        self.top_k=top_k

    def relocate(self):
        # 作成したモデルをgpu又はcpuに結び付ける。
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.to(device)

    def forward(self, h, return_features=False):
        if return_features:
            h = self.classifier.module[:3](h)
            logits = self.classifier.module[3](h)
        else:
            logits  = self.classifier(h) # K x 1
        
        y_probs = F.softmax(logits, dim = 1)
        top_instance_idx = torch.topk(y_probs[:, 1], self.top_k, dim=0)[1].view(1,)
        top_instance = torch.index_select(logits, dim=0, index=top_instance_idx)
        Y_hat = torch.topk(top_instance, 1, dim = 1)[1]
        Y_prob = F.softmax(top_instance, dim = 1) 
        results_dict = {}

        if return_features:
            top_features = torch.index_select(h, dim=0, index=top_instance_idx)
            results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs, results_dict


class MIL_fc_mc(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes = 2, top_k=1):
        super(MIL_fc_mc, self).__init__()
        assert n_classes > 2
        self.size_dict = {"small": [1024, 512]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        self.fc = nn.Sequential(*fc)

        self.classifiers = nn.ModuleList([nn.Linear(size[1], 1) for i in range(n_classes)])
        initialize_weights(self)
        self.top_k=top_k
        self.n_classes = n_classes
        assert self.top_k == 1

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc = self.fc.to(device)
        self.classifiers = self.classifiers.to(device)
    
    def forward(self, h, return_features=False):
        device = h.device
       
        h = self.fc(h)
        logits = torch.empty(h.size(0), self.n_classes).float().to(device)

        for c in range(self.n_classes):
            if isinstance(self.classifiers, nn.DataParallel):
                logits[:, c] = self.classifiers.module[c](h).squeeze(1)
            else:
                logits[:, c] = self.classifiers[c](h).squeeze(1)        

        y_probs = F.softmax(logits, dim = 1)
        m = y_probs.view(1, -1).argmax(1)
        top_indices = torch.cat(((m // self.n_classes).view(-1, 1), (m % self.n_classes).view(-1, 1)), dim=1).view(-1, 1)
        top_instance = logits[top_indices[0]]

        Y_hat = top_indices[1]
        Y_prob = y_probs[top_indices[0]]
        
        results_dict = {}

        if return_features:
            top_features = torch.index_select(h, dim=0, index=top_indices[0])
            results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs, results_dict


        
