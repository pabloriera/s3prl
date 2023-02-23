from train_utils import get_metrics
import torch
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import fire
from IPython import embed


def main(input_dir, output_dir='results'):

    dfs = []
    for record_file in sorted(list(Path(input_dir).rglob('dev*best_loss.records'))):
        records = torch.load(record_file, map_location='cpu')
        df = pd.DataFrame({k: records[k]
                          for k in ['phones', 'scores', 'labels']})
        df = df[df['phones'] != 0]
        df['labels'] = (df['labels']+1)/2
        split = None
        if record_file.stem[3].isnumeric():
            df['name'] = f'{record_file.parts[0]}_{(record_file.parts[-2])[:-2]}'
            split = record_file.stem[3]
            df['split'] = split
        else:
            df['name'] = f'{record_file.parts[0]}_{(record_file.parts[-2])}'

        dfs.append(df)

    df = pd.concat(dfs)
    df.to_csv(Path(output_dir, 'pooled_data_for_eval_dev.csv'))

    dfs = []
    for k, g in df.groupby('name'):
        metrics_table = get_metrics(g)
        metrics_table['name'] = k
        dfs.append(metrics_table)
    metrics_table_dev = pd.concat(dfs).set_index(
        'name', append=True).swaplevel(0, 1)
    metrics_table_dev.to_pickle(
        Path(output_dir, 'pooled_metrics_table_dev.pickle'))

    thrdict = {}
    for (n, k), r in metrics_table_dev.loc[(slice(None), 'all'), :].iterrows():
        ix = r['F1Score'].argmax()
        f1thr = r['F1Thr'][ix]
        thrdict[n] = f1thr

    dfs_records = []
    dfs_metrics = []
    for record_file in sorted(list(Path(input_dir).rglob('test-0.records'))):
        records = torch.load(record_file, map_location='cpu')
        df = pd.DataFrame({k: records[k]
                          for k in ['phones', 'scores', 'labels']})
        df = df[df['phones'] != 0]
        df['labels'] = (df['labels']+1)/2
        split = None
        if record_file.parts[-2][-1].isnumeric():
            name = f'{record_file.parts[0]}_{(record_file.parts[-2])[:-2]}'
            df['name'] = name
            split = str(record_file.parts[-2])[-1]
        else:
            name = f'{record_file.parts[0]}_{(record_file.parts[-2])}'
            df['name'] = name

        if split == '0' or split is None:
            dfs_records.append(df)
            metrics_table = get_metrics(df)
            metrics_table['name'] = name
            dfs_metrics.append(metrics_table)
    df = pd.concat(dfs_records)
    df.to_csv(Path(output_dir, 'pooled_data_for_eval_test.csv'))

    metrics_table_test = pd.concat(dfs_metrics).set_index(
        'name', append=True).swaplevel(0, 1)

    for (n, k), r in metrics_table_test.loc[(slice(None), 'all'), :].iterrows():
        ix = np.where(r['F1Thr'] > thrdict[n])[0][0]
        metrics_table_test.loc[(n, k), 'F1Thr'] = thrdict[n]
        metrics_table_test.loc[(n, k), 'F1Score'] = r['F1Score'][ix]

    metrics_table_test.to_pickle(
        Path(output_dir, 'pooled_metrics_table_test.pickle'))


if __name__ == '__main__':

    fire.Fire(main)
