from collections import defaultdict
import random
import numpy as np
from scipy.optimize import brentq
from scipy import interpolate
from IPython import embed
from sklearn.metrics import roc_curve, auc
import pandas as pd


import torch
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

from s3prl.downstream.runner import Runner
import torch
import joblib

import numpy as np
import pandas as pd

from IPython import embed

from torch.nn.utils.rnn import pad_sequence


def get_metrics_for_phone(df, phone, cost_fp=0.5, cost_thr=None):
    sel = df.loc[(df['phone_automatic'] == int(phone))]
    scores = np.array(sel.gop_scores)
    labels = np.array(sel.label)
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1-tpr

    # Use the best (cheating) threshold to get the min_cost
    cost_normalizer = min(cost_fp, 1.0)
    cost = (cost_fp * fpr + fnr)/cost_normalizer
    min_cost_idx = np.argmin(cost)
    min_cost_thr = thr[min_cost_idx]
    min_cost     = cost[min_cost_idx]
    min_cost_fpr = fpr[min_cost_idx]
    min_cost_fnr = fnr[min_cost_idx]

    if cost_thr is not None:
        det_pos = labels[scores>cost_thr]
        det_neg = labels[scores<=cost_thr]
        act_cost_fpr = np.sum(det_pos==0)/np.sum(labels==0)
        act_cost_fnr = np.sum(det_neg==1)/np.sum(labels==1)
        act_cost = (cost_fp * act_cost_fpr + act_cost_fnr)/cost_normalizer
#        print(min_cost, act_cost, cost_thr, min_cost_thr)
    else:
        act_cost_fpr = min_cost_fpr
        act_cost_fnr = min_cost_fnr
        act_cost = min_cost

    aucv = auc(fpr, tpr)
    eerv = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.) 

    metrics = {"1-AUC": 1-aucv,
               "EER": eerv,
               "MinCost": min_cost,
               "MinCostThr": min_cost_thr,
               "FPR4MinCost": min_cost_fpr,
               "FNR4MinCost": min_cost_fnr,
               "ActCost": act_cost,
               "FPR4ActCost": act_cost_fpr,
               "FNR4ActCost": act_cost_fnr,
               "POS_COUNT": np.sum(labels),
               "NEG_COUNT": len(labels)-np.sum(labels),
               "FPR": fpr,
               "FNR": fnr,
               "POS": scores[labels==1],
               "NEG": scores[labels==0]
    }

    return metrics


def get_metrics(df, cost_thrs=None):

    metrics = dict()
    include_phones = df['phone_automatic'].unique()

    for phone in include_phones:
        cost_thr = cost_thrs[phone]['MinCostThr'] if cost_thrs is not None else None
        metrics[phone] = get_metrics_for_phone(df, phone, cost_thr=cost_thr)
        
    # Add the mean metrics over all phones
    metrics_to_average = [m for m in metrics[include_phones[0]].keys() if m not in ["FPR", "FNR", "POS", "NEG"]]
    metrics["mean"] = dict([(m, np.mean([metrics[p][m] for p in include_phones])) for m in metrics_to_average])
    
    return metrics
        



def get_max_length(phone_ids_list):
    max_len = max(len(x) for x in phone_ids_list)
    return max_len


def get_summarisation_data(phones_array, max_len_col):
    phones_array = phones_array.cpu().detach().numpy()
    index = np.where(phones_array[:-1] != phones_array[1:])[0]
    rows = []
    phones = []
    start = 0
    for i in index:
        tmp_row = np.zeros(max_len_col)
        end = i
        tmp_row[start:end+1] = 1
        rows.append(tmp_row)
        phones.append(phones_array[i])
        start = end+1
    res = np.stack(rows, axis=0)
    return(res, phones)


def evaluate(runner, split=None, logger=None, global_step=0):
    """evaluate function will always be called on a single process even during distributed training"""
    
    # When this member function is called directly by command line
    split = runner.args.evaluate_split

    # fix seed to guarantee the same evaluation protocol across steps
    random.seed(runner.args.seed)
    np.random.seed(runner.args.seed)
    torch.manual_seed(runner.args.seed)
   
    # record original train/eval states and set all models to eval
    trainings = []
    for entry in runner.all_entries:
        trainings.append(entry.model.training)
        entry.model.eval()

    # prepare data
    dataloader = runner.downstream.model.get_dataloader(split)
    evaluate_ratio = float(runner.config["runner"].get("evaluate_ratio", 1))
    evaluate_steps = round(len(dataloader) * evaluate_ratio)

    output = []
    for batch_id, (wavs, *others) in enumerate(tqdm(dataloader, dynamic_ncols=True, desc=split, total=evaluate_steps)):
        if batch_id > evaluate_steps:
            break
        wavs = [torch.FloatTensor(wav).to(runner.args.device)
                for wav in wavs]
        with torch.no_grad():
            features = runner.upstream.model(wavs)
            features = runner.featurizer.model(wavs, features)
            labels, phone_ids = others
            features, labels, phone_ids, lengths = runner.downstream.model.process_input_forward(features, labels, phone_ids)
            predicted = runner.downstream.model.model(features)
            labels = labels*2-1
            
            max_len_col = labels.shape[1]           
            summarisation_matrix_list = []
            phones_id_list = []
 
            for element in others[1]:
                matrix, phonesids = get_summarisation_data(element, max_len_col)
                summarisation_matrix_list.append(matrix)
                phones_id_list.append(phonesids)

            max_len_row = get_max_length(summarisation_matrix_list)
            
            padded_sum_mats_list = []
            for summarisation_matrix in summarisation_matrix_list:
                npad = [(0, max_len_row-len(summarisation_matrix)), (0,0)]
                padded_summarisation_matrix = np.pad(summarisation_matrix, npad, 'constant', constant_values=0)
                padded_sum_mats_list.append(padded_summarisation_matrix)
            
            
            mats2tensor = padded_sum_mats_list[0].T
            for m in padded_sum_mats_list[1:]:
                mats2tensor = np.dstack((mats2tensor, m.T))
            summarisation_batch = torch.from_numpy(mats2tensor.T).float()
       
            mask = abs(labels)
            masked_outputs = predicted*mask
            logits_by_phone_1hot = torch.matmul(summarisation_batch, masked_outputs)
            labels_by_phone_1hot = torch.sign(torch.matmul(summarisation_batch, labels))
            frame_counts = torch.matmul(summarisation_batch, mask)
            frame_counts[frame_counts==0]=1

            mean_logits_1hot = torch.div(logits_by_phone_1hot, frame_counts)
            mean_logits_by_phone = torch.sum(mean_logits_1hot, dim=2)
            gops_by_phone = torch.log(mean_logits_by_phone).cpu().detach().numpy()
            labels_by_phone = torch.sum(labels_by_phone_1hot, dim=2).cpu().detach().numpy()

            df_batch_dict =  defaultdict(list)
            
            for i,phnlist in enumerate(phones_id_list):
                phrase_phones = []
                phrase_labels = []
                phrase_gops =  []
                for j,phone in enumerate(phnlist):
                    phrase_phones.append(phone)
                    phrase_labels.append(int(labels_by_phone[i][j]))
                    phrase_gops.append(gops_by_phone[i][j])
                df_batch_dict['phone_automatic'] += phrase_phones
                df_batch_dict['gop_scores'] += phrase_gops
                df_batch_dict['label'] += phrase_labels

            df_batch = pd.DataFrame.from_dict(df_batch_dict)
            output.append(df_batch)
    
    df2eval = pd.concat(output)
    df2eval = df2eval[df2eval['label']!=0]
    di = {-1: 0, 1:1}
    df2eval = df2eval.replace({"label": di})
    df2eval['gop_scores'] = df2eval['gop_scores'].fillna(-5)
    
    df2eval.to_pickle(Path(output_dir, output_filename))
    
    metrics_dict = get_metrics(df2eval, cost_thrs=None)
    metrics_table = pd.DataFrame.from_dict(metrics_dict).T.reset_index().rename(columns={'index':'phone_target'})
    metrics_table.to_pickle(Path(output_dir, 'metrics_table.pickle'))




from pathlib import Path

ckpt_path  = '/mnt/raid1/jazmin/exps/s3prl/s3prl/result/downstream/run9_alphaft/best-states-dev.ckpt'
output_dir = Path(ckpt_path).parent
output_filename  = 'data_for_eval.pickle'
ckpt  = torch.load(ckpt_path, map_location='cpu')
ckpt['Config']['downstream_expert']['datarc']['data_root'] = '/mnt/raid1/jazmin/data/l2arctic/'
ckpt['Args'].device    = 'cpu'
ckpt['Args'].init_ckpt = ckpt_path
split = 'test'

runner = Runner(ckpt['Args'], ckpt['Config'])
evaluate(runner, split)

