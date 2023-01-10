from collections import defaultdict
import random
import tempfile

import torch
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

from s3prl.downstream.runner import Runner
import torch
import numpy as np

from IPython import embed

def get_summarisation_data(phones_array):
    index = np.where(phones_array[:-1] != phones_array[1:])[0]
    rows = []
    frame_counts = []
    phones = []
    start = 0
    for i in index:
        tmp_row = np.zeros(len(phones))
        end = i
        tmp_row[start:end+1] = 1
        num_frames = np.sum(tmp_row)
        rows.append(tmp_row)
        frame_counts.append(num_frames)
        phones.append(phones_array[i])
        start = end+1
    res = np.stack(rows, axis=0)
    return(res, frame_counts, phones)


def evaluate(runner, split=None, logger=None, global_step=0):
    """evaluate function will always be called on a single process even during distributed training"""
    
    # When this member function is called directly by command line
    not_during_training = split is None and logger is None and global_step == 0
    if not_during_training:
        split = runner.args.evaluate_split
        tempdir = tempfile.mkdtemp()
        logger = SummaryWriter(tempdir)
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

    batch_ids = []
    records = defaultdict(list)
    for batch_id, (wavs, *others) in enumerate(tqdm(dataloader, dynamic_ncols=True, desc=split, total=evaluate_steps)):
        if batch_id > evaluate_steps:
            break
        wavs = [torch.FloatTensor(wav).to(runner.args.device)
                for wav in wavs]
        with torch.no_grad():
            features = runner.upstream.model(wavs)
            features = runner.featurizer.model(wavs, features)
            -, out, labs = runner.downstream.model(
                split,
                features, *others,
                records=records,
                batch_id=batch_id,)
            batch_ids.append(batch_id)
            
            summarization_batch = []
            frame_counts_batch = []
            phones_batch = []
            for element in others:
                matrix, frame_counts, phones = get_summarisation_data(others[1])
                summarization_batch.append(matrix)
                frame_counts_batch.append(frame_counts)
                phones_batch.append(phones)
            
            summarization_tensor = np.dstack(summarization_batch)
            summarization_tensor = torch.from_numpy(summarization_tensor)


        embed()


ckpt_path = '/mnt/raid1/jazmin/exps/s3prl/s3prl/result/downstream/run4/states-150000.ckpt'
ckpt = torch.load(ckpt_path, map_location='cpu')
ckpt['Args'].device = 'cpu'
ckpt['Args'].init_ckpt = ckpt_path
split = 'test'

runner = Runner(ckpt['Args'], ckpt['Config'])
evaluate(runner, split)

