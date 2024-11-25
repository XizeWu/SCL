import os.path as osp
from utils.util import load_file
import logging
logger = logging.getLogger('VQA') 

map_name = {'CW': 'Why', 'CH': 'How', 'TN': 'Bef&Aft', 'TC': 'When', 'DC': 'Cnt', 'DL': 'Loc', 'DO': 'Other', 'C': 'Acc_C', 'T': 'Acc_T', 'D': 'Acc_D'}

def accuracy_metric(sample_list_file, result_file):

    sample_list = load_file(sample_list_file)
    group = {'CW': [], 'CH': [], 'TN': [], 'TC': [], 'DC': [], 'DL': [], 'DO': []}
    for id, row in sample_list.iterrows():
        qns_id = str(row['video_id']) + '_' + str(row['qid'])
        qtype = str(row['type'])
        if qtype == 'TP': qtype = 'TN'
        group[qtype].append(qns_id)

    preds = result_file #load_file(result_file)
    group_acc = {'CW': 0, 'CH': 0, 'TN': 0, 'TC': 0, 'DC': 0, 'DL': 0, 'DO': 0}
    group_cnt = {'CW': 0, 'CH': 0, 'TN': 0, 'TC': 0, 'DC': 0, 'DL': 0, 'DO': 0}
    overall_acc = {'C':0, 'T':0, 'D':0}
    overall_cnt = {'C':0, 'T':0, 'D':0}
    all_acc = 0
    all_cnt = 0
    for qtype, qns_ids in group.items():
        cnt = 0
        acc = 0
        for qid in qns_ids:

            cnt += 1
            answer = preds[qid]['answer']
            pred = preds[qid]['prediction']

            if answer == pred: 
                acc += 1

        group_cnt[qtype] = cnt
        group_acc[qtype] += acc
        overall_acc[qtype[0]] += acc
        overall_cnt[qtype[0]] += cnt
        all_acc += acc
        all_cnt += cnt


    for qtype, value in overall_acc.items():
        group_acc[qtype] = value
        group_cnt[qtype] = overall_cnt[qtype]

    logger.debug("       ".join(list(map(lambda q_type: map_name[q_type], group_acc.keys()))))
    logger.debug("      ".join(['{:.2f}'.format(acc*100.0/group_cnt[qtype]) for qtype, acc in group_acc.items()]))
    logger.debug('Acc: {:.2f}'.format(all_acc*100.0/all_cnt))

    logger.debug("----------------------------------------------------------------------------------------")
    logger.debug(" ".join(['qtype={}, {}/{} \n'.format(qtype, acc, group_cnt[qtype]) for qtype, acc in group_acc.items()]))


def main(result_file, mode='val'):
    dataset_dir = ''
    data_set = mode
    sample_list_file = osp.join(dataset_dir, data_set+'.csv')
    logger.debug('Evaluating {}'.format(result_file))

    accuracy_metric(sample_list_file, result_file)


if __name__ == "__main__":
    result_file = load_file("json file path")
    accuracy_metric("csv file path", result_file)
