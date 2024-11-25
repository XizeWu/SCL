import random

import torch
import os
import h5py
import os.path as osp
import numpy as np
from torch.utils import data
from utils.util import load_file, pause, group, get_qsn_type
from torch.utils.data import Dataset
import random as rd

class VideoQADataset_NExTQA(Dataset):
    def __init__(self, mode, mc=5, cl_loss=1, csv_path="", grid_path="", app_path=""):
        super(VideoQADataset_NExTQA, self).__init__()
        # dataset
        self.sample_list_file = osp.join(csv_path, "{}.csv".format(mode))
        self.sample_list = load_file(self.sample_list_file)
        self.mode = mode
        self.mc = mc
        self.lvq = cl_loss

        # video feature
        frame_feat_file = osp.join(app_path, '{}_frames.h5'.format(mode))
        grid_feat_file = osp.join(grid_path, "{}_grid.h5".format(mode))
        print("Loading {} ...".format(frame_feat_file))
        print("Loading {} ...".format(grid_feat_file))
        self.frame_feats = {}
        self.grid_feats = {}

        # csv file
        if self.mode not in ['val', 'test']:
            self.all_answers = set(self.sample_list['answer'])
            self.all_questions = set(self.sample_list['question'])
            self.ans_group, self.qsn_group = group(self.sample_list, gt=False)

        # frame feature
        with h5py.File(frame_feat_file, "r") as fp:
            vids = fp["ids"]
            feats = fp["feats"][:, ::2, :]
            for id, (vid, feat) in enumerate(zip(vids, feats)):
                self.frame_feats[str(vid)] = feat

        # grid feature
        with h5py.File(grid_feat_file, "r") as fp:
            vids = fp["ids"]
            feats = fp["feats"][:, ::2]
            for id, (vid, feat) in enumerate(zip(vids, feats)):
                self.grid_feats[str(vid)] = feat

        print("----"*20)

    def __getitem__(self, idx):
        cur_sample = self.sample_list.iloc[idx]

        video_name = str(cur_sample['video_id'])
        q_idx = str(cur_sample['qid'])
        qus_word = str(cur_sample['question'])

        vid_frame_feat = torch.from_numpy(self.frame_feats[video_name]).type(torch.float32)
        vid_grid_feat = torch.from_numpy(self.grid_feats[video_name]).type(torch.float32)

        qus_id = video_name + '_' + q_idx

        qsn_id, qtxt_contrastive = 0, 0
        qtxt_contrastive2 = 0
        qtype = 'null' if 'type' not in cur_sample else cur_sample['type']
        if self.lvq and self.mode not in ['val', 'test']:
            qtype = get_qsn_type(qus_word, qtype)
            neg_num = 5
            if qtype not in self.qsn_group or len(self.qsn_group[qtype]) < neg_num - 1:
                valid_qsncans = self.all_questions      # self.qsn_group[self.mtype]
            else:
                valid_qsncans = self.qsn_group[qtype]

            cand_qsn = valid_qsncans - set(qus_word)
            qtxt_contrastive = rd.sample(list(cand_qsn), neg_num - 1)
            qtxt_contrastive.append(qus_word)
            rd.shuffle(qtxt_contrastive)
            qsn_id = qtxt_contrastive.index(qus_word)

            qtxt_contrastive2 = [x + " " + cur_sample['answer'] for x in qtxt_contrastive]

        ans = cur_sample['answer']
        choices = [str(cur_sample["a" + str(i)]) for i in range(self.mc)]
        answer_id = choices.index(ans) if ans in choices else -1

        if self.mode not in ['val', 'test'] and rd.random() < 0.3:
            # add randomness to negative answers
            qtype = cur_sample['type']
            if qtype == 'TP':
                qtype = 'TN'
            qtype = get_qsn_type(qus_word, qtype)  # argument 'qtype' is used to distinguish Question or Reason in CausalVid-QA

            if qtype not in self.ans_group or len(self.ans_group[qtype]) < self.mc - 1:
                valid_anscans = self.all_answers
            else:
                valid_anscans = self.ans_group[qtype]

            cand_answers = valid_anscans - set(ans)
            choices = rd.sample(list(cand_answers), self.mc - 1)
            choices.append(ans)

            rd.shuffle(choices)
            answer_id = choices.index(ans)

        ans_word = ['[CLS] ' + qus_word + ' [SEP] ' + opt for opt in choices]

        return {
            "video_frame": vid_frame_feat,
            "video_grid": vid_grid_feat,
            "qus_word": qus_word,
            "qus_id": qus_id,
            "candi_word": ans_word,
            "ans_id": answer_id,
            "type": qtype,
            "qsn_word": qtxt_contrastive2,
            "qsn_id": qsn_id,
        }

    def __len__(self):
        return len(self.sample_list)