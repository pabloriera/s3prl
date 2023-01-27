from pathlib import Path
import argparse
from collections import defaultdict
import random
import numpy as np
from scipy.optimize import brentq
from scipy import interpolate
from IPython import embed
from sklearn.metrics import roc_curve, auc
import pandas as pd
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support


import torch
import numpy as np
from tqdm import tqdm
import yaml

from s3prl.downstream.runner import Runner
import torch

import numpy as np
import pandas as pd

from IPython import embed
from downstream.pronscor_classification.dataset import PronscorDataset
from downstream.pronscor_classification.train_utils import process_input_forward, get_phones_masks, get_summarisation


def compute_metrics(df, cost_fp=0.5, cost_thr=None, f1_thr=None):
    scores = np.array(df.gop_scores)
    labels = np.array(df.label)

    if f1_thr is None:
        precision, recall, f1_thr = precision_recall_curve(
            df['label'], df['gop_scores'])

        numerator = 2 * recall * precision
        denom = recall + precision
        f1_scores = np.divide(
            numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
    else:
        precision, recall, f1_scores, _ = precision_recall_fscore_support(
            df['label'], df['gop_scores'] > f1_thr, average='binary')

    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1-tpr

    # Use the best (cheating) threshold to get the min_cost
    cost_normalizer = min(cost_fp, 1.0)
    cost = (cost_fp * fpr + fnr)/cost_normalizer
    min_cost_idx = np.argmin(cost)
    min_cost_thr = thr[min_cost_idx]
    min_cost = cost[min_cost_idx]
    min_cost_fpr = fpr[min_cost_idx]
    min_cost_fnr = fnr[min_cost_idx]

    if cost_thr is not None:
        det_pos = labels[scores > cost_thr]
        det_neg = labels[scores <= cost_thr]
        act_cost_fpr = np.sum(det_pos == 0)/np.sum(labels == 0)
        act_cost_fnr = np.sum(det_neg == 1)/np.sum(labels == 1)
        act_cost = (cost_fp * act_cost_fpr + act_cost_fnr)/cost_normalizer
#        print(min_cost, act_cost, cost_thr, min_cost_thr)
    else:
        act_cost_fpr = min_cost_fpr
        act_cost_fnr = min_cost_fnr
        act_cost = min_cost

    aucv = auc(fpr, tpr)
    eerv = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)

    metrics = {
        "1-AUC": 1-aucv,
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
        "POS": scores[labels == 1],
        "NEG": scores[labels == 0],
        "Recall": recall,
        "Precision": precision,
        "F1Score": f1_scores,
        "F1Thr": f1_thr
    }

    return metrics


def get_metrics(df, cost_thrs=None, f1_thr=None):

    metrics = dict()

    metrics['all'] = compute_metrics(df, cost_thr=None, f1_thr=f1_thr)

    for phone, g in df.groupby('phone_automatic'):
        cost_thr = cost_thrs[phone]['MinCostThr'] if cost_thrs is not None else None
        metrics[phone] = compute_metrics(g, cost_thr=cost_thr, f1_thr=f1_thr)

    metrics_table = pd.DataFrame(metrics).T
    metrics_table.index.name = 'phone_automatic'

    return metrics_table


def evaluate(runner, split=None, phone_db_map=None, silence_id=0):
    """evaluate function will always be called on a single process even during distributed training"""

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
            labels, phone_ids = others.copy()

            num_phones = dataloader.dataset.class_num

            features, labels, phone_ids, lengths = process_input_forward(
                features, labels, phone_ids, num_phones, silence_id=silence_id)
            
            predicted = runner.downstream.model.model(features)


            if phone_db_map is not None:
                predicted = predicted[:, :, phone_db_map['predicted'].tolist()]
                labels = labels[:, :,
                                phone_db_map['labels'].tolist()]
     
            
            
            logits_by_phone_1hot, labels_by_phone_1hot, frame_counts, phones_id_list = get_summarisation(phone_ids, labels, predicted, summarise)
            

            if summarisation == 'lpp':
                mean_logits_1hot = torch.div(logits_by_phone_1hot, frame_counts)
                gops_by_phone = torch.sum(mean_logits_1hot, dim=2)
            else:
                gops_by_phone = torch.sum(logits_by_phone_1hot, dim=2)
            
            
            labels_by_phone = torch.sum(labels_by_phone_1hot, dim=2)

            df_batch_dict = defaultdict(list)

            for i, (phnlist, labs, gops) in enumerate(zip(phones_id_list, labels_by_phone, gops_by_phone)):
                phrase_labels = labs[:len(phnlist)]
                phrase_gops = gops[:len(phnlist)]
                df_batch_dict['phone_automatic'] += phnlist.tolist()
                df_batch_dict['gop_scores'] += phrase_gops.tolist()
                df_batch_dict['label'] += phrase_labels.tolist()

            df_batch = pd.DataFrame(df_batch_dict)
            assert df_batch[df_batch['phone_automatic']
                            == 0]['gop_scores'].sum() == 0
            output.append(df_batch)

    df_scores = pd.concat(output)
    df_scores = df_scores[df_scores['phone_automatic'] != 0]
    df_scores['label'] = (df_scores['label']+1)/2

    assert (df_scores['label'] == 0.5).sum() == 0

    return df_scores


def main(ckpt_path, splits, config=None, phone_db_map=None, output_dir=None):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        torch.multiprocessing.set_sharing_strategy('file_system')

    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt['Args'].device = device
    ckpt['Args'].init_ckpt = ckpt_path

    runner = Runner(ckpt['Args'], ckpt['Config'])

    silence_id = 0
    if phone_db_map is not None:
        phone_db_map_df = pd.read_csv(
            'downstream/pronscor_classification/phone-db-map.csv')

        phone_db_map_df = phone_db_map_df[[phone_db_map[0], phone_db_map[1]]].applymap(
            lambda x: int(x) if x.isnumeric() else np.nan).dropna().astype(int)
        phone_db_map = {'predicted': phone_db_map_df[phone_db_map[0]],
                        'labels': phone_db_map_df[phone_db_map[1]],
                        }
        # TODO: silence_id
        assert False

    dev_best_f1_thr = None
    for split in splits:
        print("Split", split)

        if config is not None:
            with open(config, 'r') as fp:
                config_ = yaml.safe_load(fp)
            datarc = config_['downstream_expert']['datarc']
            datarc['merge_phones'] = {4: 1}
            ds = PronscorDataset(split, datarc['eval_batch_size'], **datarc)
            setattr(runner.downstream.model, f'{split}_dataset', ds)

        df_scores = evaluate(
            runner, split, phone_db_map=phone_db_map, silence_id=silence_id)

        if split == 'dev':
            metrics_table = get_metrics(
                df_scores, cost_thrs=None, f1_thr=None)
            ix = np.nanargmax(metrics_table.loc['all']['F1Score'])
            dev_best_f1_thr = metrics_table.loc['all']['F1Thr'][ix]
            print('Dev Best threshold: ', dev_best_f1_thr)
            print('Dev Best F1-Score: ',
                  np.nanmax(metrics_table.loc['all']['F1Score']))
        elif split == 'test':
            metrics_table = get_metrics(
                df_scores, cost_thrs=None, f1_thr=dev_best_f1_thr)

            print('Test F1-Score: ', metrics_table.loc['all']['F1Score'])

        output_filename = f'data_for_eval_{split}.pickle'

        if output_dir is None:
            output_dir = Path(ckpt_path).parent
        elif not Path(output_dir).exists():
            Path(output_dir).mkdir(exist_ok=True)
        df_scores.to_pickle(Path(output_dir, output_filename))

        output_filename = f'metrics_table_{split}.pickle'
        print("Output", output_dir)
        print(metrics_table.loc['all'].to_string())
        metrics_table.to_pickle(Path(output_dir, output_filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--ckpt', required=True)
    parser.add_argument(
        '-s', '--split',  nargs="+",  choices=['train', 'test', 'dev'], required=True)
    parser.add_argument(
        '--config', default=None)
    parser.add_argument(
        '--phone-db-map', nargs="+",  default=None)
    parser.add_argument(
        '--output-dir', default=None)

    args = parser.parse_args()

    main(args.ckpt, args.split, args.config,
         args.phone_db_map, args.output_dir)
