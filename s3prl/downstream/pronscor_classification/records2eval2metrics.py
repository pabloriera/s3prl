import argparse
from train_utils import get_metrics
import torch
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def main(results, split):
    for record_file in Path(results).rglob(f'{split}*.records'):
        records = torch.load(record_file)
        df = pd.DataFrame({k: records[k]
                          for k in ['phones', 'scores', 'labels']})
        df = df[df['phones'] != 0]
        df['labels'] = (df['labels']+1)/2
        df.to_pickle(Path(record_file.parent, f'data_for_eval_{split}.pickle'))
        metrics_table = get_metrics(df, cost_thrs=None)
        metrics_table.to_pickle(
            Path(record_file.parent, f'metrics_table_{split}.pickle'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-r', '--results', required=True)
    parser.add_argument(
        '-t', '--split', default="test")

    args = parser.parse_args()

    main(args.results, args.split)
