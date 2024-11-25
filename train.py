# from torch.nn.modules.module import _IncompatibleKeys
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
from utils.util import EarlyStopping, save_file, set_gpu_devices, pause, set_seed
from utils.logger import logger
import eval_mc
import time
import logging
import argparse
import os.path as osp
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description="train parameter")
# general
parser.add_argument("-v", type=str, default="IRCC", help="version")
parser.add_argument("-bs", type=int, action="store", help="BATCH_SIZE8", default=32)
parser.add_argument("-lr", type=float, action="store", help="learning rate", default=1e-5)
parser.add_argument("-epoch", type=int, action="store", help="epoch for train", default=30)
parser.add_argument("-gpu", type=int, help="set gpu id", default=0)
parser.add_argument("-es", action="store_true", help="early_stopping")
parser.add_argument("-dropout", "-drop", type=float, help="dropout rate", default=0.1)
parser.add_argument("-encoder_dropout", "-ep", type=float, help="dropout rate", default=0.1)   
parser.add_argument("-patience", "-pa", type=int, help="patience of ReduceonPleatu", default=1)
parser.add_argument("-gamma", "-ga", type=float, help="gamma of MultiStepLR", default=0.5)
parser.add_argument("-decay", type=float, help="weight decay", default=0.001) 

# dataset
parser.add_argument('-dataset', default='next-qa', choices=['next-qa'], type=str)
parser.add_argument("-max_word_q", default=30, type=int)
parser.add_argument("-max_word_qa", default=37, type=int)
parser.add_argument("-theta", default=0.2, type=float)

# path list
parser.add_argument('-csv_path', default='the csv file path', type=str)
parser.add_argument('-grid_path', default='the grid feature', type=str)
parser.add_argument('-app_path', default='the frame feature', type=str)

parser.add_argument('-G', default=6, type=int)
parser.add_argument('-numF', default=16, type=int)


# model
parser.add_argument("-d_model", "-md",  type=int, help="hidden dim of vq encoder", default=768) 
parser.add_argument("-word_dim", "-wd", type=int, help="word dim ", default=768)   
parser.add_argument("-hard_eval", "-hd", default="True", help="hard selection during inference")

# transformer
parser.add_argument("-num_encoder_layers", "-el", type=int, help="number of encoder layers in transformer", default=1)
parser.add_argument("-num_decoder_layers", "-dl", type=int, help="number of decoder layers in transformer", default=1)
parser.add_argument("-n_query", type=int, help="num of query", default=5) 
parser.add_argument("-nheads", type=int, help="num of attention head", default=8) 
parser.add_argument("-normalize_before", action="store_true", help="pre or post normalize")
parser.add_argument("-activation", default='relu', choices=['relu', 'gelu', 'glu'], type=str)

# lan model
parser.add_argument("-text_encoder_lr", "-tlr", type=float, action="store", help="learning rate for lan model", default=5e-6)
parser.add_argument("-freeze_text_encoder", action="store_true", help="freeze text encoder")
parser.add_argument("-text_encoder_type", "-t", default="microsoft/deberta-base", type=str)
parser.add_argument('-text_pool_mode', "-pool", default=0, choices=[0, 1, 2], help="0last hidden, 1mean, 2max", type=int)

# cl
parser.add_argument("-a", type=float, action="store", help="NCE loss multiplier", default=1)
parser.add_argument("-cl_loss", type=float, help="contrastive loss (Q+ and Q-)", default=0.01)

args = parser.parse_args()
set_gpu_devices(args.gpu)
set_seed(999)
set_gpu_devices(args.gpu)



import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from networks.model import VideoQAmodel
from DataLoader import VideoQADataset_NExTQA

# from torch.utils.tensorboard import SummaryWriter
torch.set_printoptions(linewidth=200)
np.set_printoptions(edgeitems=30, linewidth=30, formatter=dict(float=lambda x: "%.3g" % x))
# torch.autograd.set_detect_anomaly(True)

def train(model, optimizer, train_loader, xe, device):
    model.train()
    total_step = len(train_loader)
    epoch_loss = 0.0
    prediction_list = []
    epoch_losscl, epoch_lossvqa = 0.0, 0.0
    answer_list = []
    for iter, inputs in enumerate(tqdm(train_loader)):
        vid_frame_feat, vid_grid_feat = inputs["video_frame"].to(device), inputs["video_grid"].to(device)
        qus_word, qus_id = inputs["qus_word"], inputs["qus_id"]
        candi_word, ans_targets = inputs["candi_word"], inputs["ans_id"].to(device)
        qsn_word, qsn_targets = inputs["qsn_word"], inputs["qsn_id"].to(device)

        with torch.cuda.amp.autocast(enabled=True):
            out_f, ans_g = model(vid_frame_feat, vid_grid_feat, qus_word, ans_word=candi_word, ans_id=None, stage="GQA")
            vt_proj = out_f.unsqueeze(2)
            vq_predicts = torch.bmm(ans_g, vt_proj).squeeze()  # [bs, 5]
            loss_vqa = xe(vq_predicts, ans_targets)

            if args.cl_loss:
                vq_feat, qsn_faet = model(vid_frame_feat, vid_grid_feat, qus_word, ans_word=qsn_word, ans_id=None, stage="GD")
                vq_feat = vq_feat.unsqueeze(2)
                cl_predicts = torch.bmm(qsn_faet, vq_feat).squeeze()  # [bs, 5]
                loss_cl = args.cl_loss * xe(cl_predicts, qsn_targets)

                loss = loss_cl + loss_vqa
            else:
                loss = loss_vqa

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        if args.cl_loss:
            epoch_losscl += loss_cl.item()
        epoch_lossvqa += loss_vqa.item()
        prediction = vq_predicts.max(-1)[1]  # bs,
        prediction_list.append(prediction)
        answer_list.append(ans_targets.cpu())

    predict_answers = torch.cat(prediction_list, dim=0).long().cpu()
    ref_answers = torch.cat(answer_list, dim=0).long()
    acc_num = torch.sum(predict_answers == ref_answers).numpy()

    return epoch_loss / total_step, epoch_lossvqa / total_step, epoch_losscl / total_step, acc_num * 100.0 / len(
        ref_answers)

def eval(model, val_loader, device):
    model.eval()
    prediction_list = []
    answer_list = []
    with torch.no_grad():
        for iter, inputs in enumerate(val_loader):
            vid_frame_feat, vid_grid_feat = inputs["video_frame"].to(device), inputs["video_grid"].to(device)
            qus_word, qus_id = inputs["qus_word"], inputs["qus_id"]
            candi_word, ans_id = inputs["candi_word"], inputs["ans_id"]

            out_f, ans_g = model(vid_frame_feat, vid_grid_feat, qus_word, ans_word=candi_word, ans_id=None, stage="GQA")
            fusion_proj = out_f.unsqueeze(2)
            predicts = torch.bmm(ans_g, fusion_proj).squeeze()

            prediction = predicts.max(-1)[1]   # bs,
            prediction_list.append(prediction)
            answer_list.append(ans_id)

    predict_answers = torch.cat(prediction_list, dim=0).long().cpu()
    ref_answers = torch.cat(answer_list, dim=0).long()
    acc_num = torch.sum(predict_answers==ref_answers).numpy()

    return acc_num*100.0 / len(ref_answers)


def predict(model,test_loader, device):
    """
    predict the answer with the trained model
    :param model_file:
    :return:
    """

    model.eval()
    results = {}
    prediction_list = []
    answer_list = []
    with torch.no_grad():
        for iter, inputs in enumerate(test_loader):
            vid_frame_feat, vid_grid_feat = inputs["video_frame"].to(device), inputs["video_grid"].to(device)
            qus_word, qus_id = inputs["qus_word"], inputs["qus_id"]
            candi_word, ans_id = inputs["candi_word"], inputs["ans_id"]

            out_f, ans_g = model(vid_frame_feat, vid_grid_feat, qus_word, ans_word=candi_word, ans_id=None, stage="GQA")
            fusion_proj = out_f.unsqueeze(2)
            predicts = torch.bmm(ans_g, fusion_proj).squeeze()

            prediction = predicts.max(-1)[1]    # bs,
            prediction_list.append(prediction)
            answer_list.append(ans_id)

            for qid, pred, ans in zip(qus_id, prediction.data.cpu().numpy(), ans_id.numpy()):
                results[qid] = {'prediction': int(pred), 'answer': int(ans)}
    
    predict_answers = torch.cat(prediction_list, dim=0).long().cpu()
    ref_answers = torch.cat(answer_list, dim=0).long()
    acc_num = torch.sum(predict_answers==ref_answers).numpy()

    return results, acc_num*100.0 / len(ref_answers)


if __name__ == "__main__":

    # writer = SummaryWriter('./log/tensorboard')
    logger, sign = logger(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = VideoQADataset_NExTQA(mode='train', mc=args.n_query, cl_loss=args.cl_loss, csv_path=args.csv_path,
                                          grid_path=args.grid_path, app_path=args.app_path)
    val_dataset = VideoQADataset_NExTQA(mode='val', mc=args.n_query, cl_loss=args.cl_loss, csv_path=args.csv_path,
                                        grid_path=args.grid_path, app_path=args.app_path)
    test_dataset = VideoQADataset_NExTQA(mode='test', mc=args.n_query, cl_loss=args.cl_loss, csv_path=args.csv_path,
                                         grid_path=args.grid_path, app_path=args.app_path)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True,
                              prefetch_factor=4)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.bs, shuffle=False, num_workers=8, pin_memory=True,
                            prefetch_factor=4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=8, pin_memory=True,
                             prefetch_factor=4)

    # hyper setting
    epoch_num = args.epoch
    args.device = device
    config = {**vars(args)}
    model = VideoQAmodel(**config)

    for name, param in model.named_parameters():
        if "word_embeddings" in name:
            param.requires_grad = False

    param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad]}]
    optimizer = torch.optim.AdamW(params=param_dicts, lr=args.lr, weight_decay=args.decay)
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=args.gamma, patience=args.patience, verbose=True)
    model.to(device)
    xe = nn.CrossEntropyLoss().to(device)

    # Creates a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # train & val
    best_eval_score = 0.0
    best_epoch = 1
    for epoch in range(1, epoch_num + 1):
        train_loss, loss_vqa, loss_cl, train_acc = train(model, optimizer, train_loader, xe, device)
        val_acc = eval(model, val_loader, device)
        scheduler.step(val_acc)
        if val_acc > best_eval_score:
            best_eval_score = val_acc
            best_epoch = epoch
            best_model_path = './models/best_model-{}.ckpt'.format(sign)
            torch.save(model.state_dict(), best_model_path)

        test_acc = eval(model, test_loader, device)

        logger.debug(
            "==>Epoch:[{}/{}][LR{}][Loss: {:.4f}, intra: {:.4f}, inter: {:.4f} | TrACC: {:.2f} ValACC: {:.2f} TeACC: {:.2f}]".
            format(epoch, epoch_num, optimizer.param_groups[0]['lr'], train_loss, loss_vqa, loss_cl, train_acc, val_acc, test_acc))

    logger.debug("Epoch {} Best Val acc{:.2f}".format(best_epoch, best_eval_score))

    # predict with best model
    model.load_state_dict(torch.load(best_model_path))
    results, test_acc = predict(model, test_loader, device)
    logger.debug("Test acc{:.2f} on {} epoch".format(test_acc, best_epoch))

    eval_mc.accuracy_metric(test_dataset.sample_list_file, results)
    result_path = './prediction/{}-{}-{:.2f}.json'.format(sign, best_epoch, best_eval_score)
    save_file(results, result_path)