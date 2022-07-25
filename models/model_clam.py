import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        # self.moduleは全結合層とTanh関数のリスト 
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]
        # dropoutがTrueならself.moduleに25%のDropoutを追加
        if dropout:
            self.module.append(nn.Dropout(0.25))

        # self.moduleに隠れ層(256次元)から出力1への全結合層を追加
        self.module.append(nn.Linear(D, n_classes))
        # 上記のリストからself.moduleモデルを作成        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):#
        # 上記のself.moduleモデルにxを入力した結果と、xを出力
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        # self.attention_aは全結合層とTanh関数のリスト 
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        # self.attention_bは全結合層とsigmoid関数のリスト         
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        # dropoutがTrueならself.attention_a,_bに25%のDropoutを追加
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        # 上記のリストからself.attention_a,_bモデルを作成        
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        # 全結合層で隠れ層(256)から出力層(1)へ繋ぐself.attention_cモデルを作成                
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        # Aはaとbの要素積？(類似度計算、コンテキストベクトル？)
        A = a.mul(b)
        # Aにattention_cの全結合層を接続
        A = self.attention_c(A)  # N x n_classes
        return A, x

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""
# core_utils.pyのtrain関数から呼び出される。
# Sequentialでない複雑なモデルを構築するために、
# torch.nn.ModuleのサブクラスとしてCLAM_SBクラスを定義する。
class CLAM_SB(nn.Module):
    # コンストラクタ
    # main.py->core_utils.pyのtrain関数でインスタンスmodel作成時に呼び出される。
    # gate=True, 引数sizeの初期値はsmall,dropout の初期値はFalse
    # k_sample(clamのサンプルとなるpositive/negativeパッチの数)初期値は8
    # n_classesの初期値は2。
    # instance_lossの初期値はクロスエントロピー誤差、subtypingの初期値はFalse
    def __init__(self, gate = True, size_arg = "small", dropout = False, k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
        # super:親クラスのメソッドなどを呼び出すことができる。ここはpython2系の書き方。
        # nn.Moduleから継承？CLAM_SBがnn.Moduleのオブジェクト？参考:https://note.nkmk.me/python-pytorch-module-sequential/
        super(CLAM_SB, self).__init__()
        # size_dict辞書でsmallとbigの定義をしている
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        # size変数に[1024,512,256]か[1024,512,384]のどちらかを引数size_arg(small or big)に応じて入れる。
        size = self.size_dict[size_arg]
        # fc変数は[nn.Linear(1024,512),nn.ReLU()]のリスト。
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        # dropoutがtrueなら、fcリストの第三項に25%のドロップアウト層を加える。
        if dropout:
            fc.append(nn.Dropout(0.25))
        # gateがtrueなら、attention_netを本ファイル上部で定義されたAttn_Net_Gatedクラスのインスタンスにする。デフォルトはTrue。
        # 引数はLは512,Dはチャンネル数(256 or 384),dropoutはTrue or False,クラス数1,
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        # gateがtrueでないなら、attention_netを以下に定義されたAttn_Netクラスのインスタンスにする。
        # 引数はLは512,Dはチャンネル数(256 or 384),dropoutはTrue or False,クラス数1,
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        # fcに上記インスタンスattention_netを追加する。
        fc.append(attention_net)
        # self.attention_netにnn.Sequential(*fc)を代入。Sequential(*fc)で
        # fcリストに定義したネットワークがself.attention_netという名のモデルとして形成される。
        # *(アスタリスク)でリストを引数として読み込むことができる。
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        # instance_classifiersはリストに同じ全結合層をn_classesの数だけ並べる形
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        # ModuleListに全結合層のリストを渡すことで、層のイテレータを作る。
        # for i in self.instance_classifiersでprint(i)をすると、n_classesの数だけ
        # nn.Linear(512,2)の全結合層を繰り返す。
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        initialize_weights(self)

    # モデルをcuda(gpu)又はcpuに結び付ける。
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #同クラスforwardメソッドから呼び出される。つまり"CLAM_SBのインスタンス"()で__call__が呼ばれ、そこから実行されると思われる。
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        # 確認用
        print("A:",A[:10])
        print("Aの形状:",A.shape)
        #topkでは上位k個を取り出し、大きい順に値とインデックスを返す。
        #[1][-1]を指定しているということは、インデックスだけを返しているということになる。
        #top_p_idsは上位k個の内、インデックスのテンソルとなる。(例：k=2, 1位が3番目、2位が2番目にある場合=> tensor([3,2]))
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        # 確認用
        print("top_p_ids:",top_p_ids)
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        device = h.device
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
                
        M = torch.mm(A, h) 
        logits = self.classifiers(M)
        # 恐らく、(2値の内)一番値の大きい要素のインデックスを返す。[1]でインデックスを返している。[0]の場合は値を返す。
        # つまり、logitsは分類した結果のsoftmaxにかける直前なので、Y_hatは予測値として"腫瘍"か"正常"か値が大きい方のインデックスを返しているはず。
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict

class CLAM_MB(CLAM_SB):
    def __init__(self, gate = True, size_arg = "small", dropout = False, k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
        nn.Module.__init__(self)
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)] #use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        initialize_weights(self)

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        device = h.device
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h) 
        logits = torch.empty(1, self.n_classes).float().to(device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict
