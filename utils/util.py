import json
import torch
import random
import os
import os.path as osp
import re
import numpy as np
import pickle as pkl
import pandas as pd

def set_gpu_devices(gpu_id):
    gpu = ''
    if gpu_id != -1:
        gpu = str(gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu


def load_file(file_name):
    annos = None
    if osp.splitext(file_name)[-1] == '.csv':
        return pd.read_csv(file_name)
    with open(file_name, 'r') as fp:
        if osp.splitext(file_name)[1]== '.txt':
            annos = fp.readlines()
            annos = [line.rstrip() for line in annos]
        if osp.splitext(file_name)[1] == '.json':
            annos = json.load(fp)

    return annos

def save_file(obj, filename):
    """
    save obj to filename
    :param obj:
    :param filename:
    :return:
    """
    filepath = osp.dirname(filename)
    if filepath != '' and not osp.exists(filepath):
        os.makedirs(filepath)
    else:
        with open(filename, 'w') as fp:
            json.dump(obj, fp, indent=4)

def pkload(file):
    data = None
    if osp.exists(file) and osp.getsize(file) > 0:
        with open(file, 'rb') as fp:
            data = pkl.load(fp)
        # print('{} does not exist'.format(file))
    return data


def pkdump(data, file):
    dirname = osp.dirname(file)
    if not osp.exists(dirname):
        os.makedirs(dirname)
    with open(file, 'wb') as fp:
        pkl.dump(data, fp)

# split string using multi delimiters
def multisplit(delimiters, string, maxsplit=0):
    regexPattern = '|'.join(map(re.escape, delimiters))
    return re.split(regexPattern, string, maxsplit)

def pause():
    programPause=input('press to continue...')

# #################### addition ##############################
def group(csv_data, gt=True):
    ans_group, qsn_group = {}, {}
    for idx, row in csv_data.iterrows():
        qsn, ans = row['question'], row['answer']
        if gt:
            type = row['type']
            if type == 'TP': type = 'TN'
        else:
            type = 'null' if 'type' not in row else row['type']
            type = get_qsn_type(qsn, type)
        if type not in ans_group:
            ans_group[type] = {ans}
            qsn_group[type] = {qsn}
        else:
            ans_group[type].add(ans)
            qsn_group[type].add(qsn)
    return ans_group, qsn_group

def get_qsn_type(qsn, ans_rsn):
    dos = ['does', 'do', 'did']
    bes = ['was', 'were', 'is', 'are']
    w5h1 = ['what', 'who', 'which', 'why', 'how', 'where']
    qsn_sp = qsn.split()
    type = qsn_sp[0].lower()
    if type == 'what':
        if qsn_sp[1].lower() in dos:
            type = 'whata'
        elif qsn_sp[1].lower() in bes:
            type = 'whatb'
        else:
            type = 'whato'
    elif type == 'how':
        if qsn_sp[1].lower() == 'many':
            type = 'howm'
    elif type not in w5h1:
        type = 'other'
    if ans_rsn in ['pr', 'cr']:
        type += 'r'
    return type
# #################### addition ##############################

class EarlyStopping():
    """
    Early stopping to stop the training when the acc does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when acc is
               not improving
        :param min_delta: minimum difference between new acc and old acc for
               new acc to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = None
        self.early_stop = False
    def __call__(self, val_acc):
        if self.best_acc == None:
            self.best_acc = val_acc
        elif self.best_acc - val_acc < self.min_delta:
            self.best_acc = val_acc
            self.counter = 0
        elif self.best_acc - val_acc > self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False