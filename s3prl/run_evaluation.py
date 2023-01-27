from pathlib import Path
import argparse
from collections import defaultdict
import random
import numpy as np
from IPython import embed
import pandas as pd


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
from downstream.pronscor_classification.train_utils import process_input_forward, compute_metrics, get_summarisation


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
    summarise = runner.config["downstream_expert"]['datarc'].get("summarize", None))
    evaluate_steps = round(len(dataloader) * evaluate_ratio)

    output_per_frame = []
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

            # scores_tab = []
            # phones_tab = []
            # labels_tab = []
            # for i in [-1, 1]:
            #     _scores = predicted[labels == i]
            #     scores_tab.extend(_scores.tolist())
            #     labels_tab.extend(((i*torch.ones(len(_scores))+1)/2).tolist())
            #     phones_tab.extend(torch.where(labels == i)[2].tolist())

            # data_per_frame = pd.DataFrame(
            #     {'gop_scores': scores_tab,
            #      'phone_automatic': phones_tab,
            #      'label': labels_tab
            #      })

            # if phone_db_map is not None:
            #     predicted = predicted[:, :, phone_db_map['predicted'].tolist()]
            #     labels = labels[:, :,
            #                     phone_db_map['labels'].tolist()]
     
            
            logits_by_phone_1hot, labels_by_phone_1hot, frame_counts, phones_id_list = get_summarisation(phone_ids, labels, predicted, summarise)

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
            output_per_frame.append(data_per_frame)

    df_scores = pd.concat(output)
    df_scores_per_frame = pd.concat(output_per_frame)
    df_scores = df_scores[df_scores['phone_automatic'] != 0]
    df_scores['label'] = (df_scores['label']+1)/2

    assert (df_scores['label'] == 0.5).sum() == 0

    return df_scores, df_scores_per_frame


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
            print('Using config file', config)
            with open(config, 'r') as fp:
                config_ = yaml.safe_load(fp)
            datarc = config_['downstream_expert']['datarc']
            print('Using config data', datarc)
            datarc['merge_phones'] = {4: 1}
            ds = PronscorDataset(split, datarc['eval_batch_size'], **datarc)
            setattr(runner.downstream.model, f'{split}_dataset', ds)

        df_scores_per_phone, df_scores_per_frame = evaluate(
            runner, split, phone_db_map=phone_db_map, silence_id=silence_id)

        df_scores = df_scores_per_phone

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
