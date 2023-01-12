from collections import defaultdict
import random
import tempfile

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

    output = []
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
            _ , out, labs = runner.downstream.model(
                split,
                features, *others,
                records=records,
                batch_id=batch_id,)
            batch_ids.append(batch_id)
            
            max_len_col = labs.shape[1]           
            summarisation_matrix_list = []
            phones_id_list = []
            # element es phones array que es una sola frase o sea solo una tira
            # quiero convertir esa tira en una matriz
            for element in others[1]:
                # matrix deberia ser una matriz de #phones_id x max_len del batch
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
            summarisation_batch = torch.from_numpy(mats2tensor.T)
            
            mask = abs(labs)
            masked_outputs = out*mask
            logits_by_phone_1hot = torch.matmul(summarisation_batch, masked_outputs)
            labels_by_phone_1hot = torch.sign(torch.matmul(summarisation_batch, labs))
            embed()
            frame_counts = torch.matmul(summarisation_batch, mask)
            frame_counts[frame_counts==0]=1

            #le falta el log 
            mean_logits_1hot = torch.div(logits_by_phone_1hot, frame_counts)
            #gops_by_phone = np.log(torch.sum(mean_logits_1hot, dim=2).cpu().detach().numpy())
            gops_by_phone = torch.sum(mean_logits_1hot, dim=2).cpu().detach().numpy()
            labels_by_phone = torch.sum(labels_by_phone_1hot, dim=2).cpu().detach().numpy()


            df_batch_dict =  defaultdict(list)
            
            for i,phnlist in enumerate(phones_id_list):
                phrase_phones = []
                phrase_labels = []
                phrase_gops =  []
                for j,phone in enumerate(phnlist):
                    phrase_phones.append(phone)
                    phrase_labels.append(labels_by_phone[i][j])
                    phrase_gops.append(gops_by_phone[i][j])
                df_batch_dict['phone_automatic'] += phrase_phones
                df_batch_dict['gop_scores'] += phrase_gops
                df_batch_dict['label'] += phrase_labels

            df_batch = pd.DataFrame.from_dict(df_batch_dict)
            output.append(df_batch)
    
    df2eval = pd.concat(output)
    joblib.dump(df2eval, output_dir + output_filename)


# 
ckpt_path  = '/mnt/raid1/jazmin/exps/s3prl/s3prl/result/downstream/run4/states-150000.ckpt'
output_dir = '/mnt/raid1/jazmin/exps/s3prl/s3prl/result/downstream/run4'
output_filename  = 'data4eval.pickle'
ckpt       = torch.load(ckpt_path, map_location='cpu')
ckpt['Args'].device    = 'cpu'
ckpt['Args'].init_ckpt = ckpt_path
split = 'test'

runner = Runner(ckpt['Args'], ckpt['Config'])
evaluate(runner, split)

