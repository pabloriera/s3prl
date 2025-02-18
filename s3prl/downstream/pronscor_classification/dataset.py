###############
# IMPORTATION #
###############
import os
from IPython import embed
#-------------#
import pandas as pd
#-------------#
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
#-------------#
import torchaudio
import numpy as np
from pathlib import Path

# TODO: use phone dictionaries not fixed number
from .train_utils import get_phone_dictionaries, NUM_PHONES


HALF_BATCHSIZE_TIME = 8000
TEST_SPEAKERS = []

#################
# Phone Dataset #
#################


class PronscorDataset(Dataset):

    def __init__(
            self,
            split,
            bucket_size,
            data_root,
            alignments_path,
            splits_path,
            bucket_file,
            sample_rate=16000,
            bucketing=True,
            ** kwargs):

        super(PronscorDataset, self).__init__()

        self.data_root = data_root
        self.alignments_path = alignments_path
        self.splits_path = splits_path
        self.sample_rate = sample_rate
        self.class_num = NUM_PHONES  # NOTE: pre-computed, should not need change
        # Create phone dictionaries
        # self._phone_sym2int_dict, self.phone_int2sym_dict, self.phone_int2node_dict = get_phone_dictionaries(phones_list_path)

        self.Y = {}
        phone_file = open(os.path.join(
            alignments_path, 'converted_aligned_phones.txt')).readlines()
        for line in phone_file:
            line = line.strip('\n').split(' ')
            self.Y[line[0]] = [int(p) for p in line[1:]]

        self.L = {}
        label_file = open(os.path.join(
            alignments_path, 'converted_aligned_labels.txt')).readlines()
        for line in label_file:
            line = line.strip('\n').split(' ')
            self.L[line[0]] = [int(l) for l in line[1:]]

        split_list = open(os.path.join(
            splits_path, f'{split}_split.txt')).readlines()
        usage_list = [line.strip('\n') for line in split_list]

        usage_list = {line.strip('\n'): None for line in usage_list}
        print('[Dataset] - # phone classes: ' + str(self.class_num) +
              ', number of data for ' + split + ': ' + str(len(usage_list)))
        # Read table for bucketing
        assert os.path.isdir(
            bucket_file), f'Missing {bucket_file} Please first run `preprocess/generate_len_for_bucket.py` to get bucket file.'

        table = pd.read_csv(os.path.join(bucket_file, '16k.csv')).sort_values(
            by=['length'], ascending=False)

        X = table['file_path'].tolist()
        X_lens = table['length'].tolist()
        self.X = []
        self.bucketing = bucketing
        if self.bucketing:
            # Use bucketing to allow different batch sizes at run time

            batch_x, batch_len = [], []

            for x, x_len in zip(X, X_lens):
                if self._parse_x_name(x) in usage_list:
                    batch_x.append(x)
                    batch_len.append(x_len)
                    # Fill in batch_x until batch is full
                    if len(batch_x) == bucket_size:
                        # Half the batch size if seq too long
                        if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME):
                            self.X.append(batch_x[:bucket_size//2])
                            self.X.append(batch_x[bucket_size//2:])
                        else:
                            self.X.append(batch_x)
                        batch_x, batch_len = [], []

            # Gather the last batch
            if len(batch_x) > 1:
                if (bucket_size >= 2) and (len(batch_x) > bucket_size//2) and (max(batch_len) > HALF_BATCHSIZE_TIME):
                    self.X.append(batch_x[:bucket_size//2])
                    self.X.append(batch_x[bucket_size//2:])
                else:
                    self.X.append(batch_x)

            print('Batchs length')
            print(pd.DataFrame(list(map(len, self.X))).value_counts())
        else:
            for x, x_len in zip(X, X_lens):
                if self._parse_x_name(x) in usage_list:
                    self.X.append(x)

    def _parse_x_name(self, x):
        return '-'.join(x.split('.')[0].split('/')[1:])

    def _load_wav(self, wav_path):
        wav, sr = torchaudio.load(os.path.join(self.data_root, wav_path))
        assert sr == self.sample_rate, f'Sample rate mismatch: real {sr}, config {self.sample_rate}'
        return wav.view(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):

        if self.bucketing:
            # Load acoustic feature and pad
            wav_batch = [self._load_wav(x_file) for x_file in self.X[index]]
            label_batch = [torch.LongTensor(
                self.L[self._parse_x_name(x_file)]) for x_file in self.X[index]]
            phoneid_batch = [torch.LongTensor(
                self.Y[self._parse_x_name(x_file)]) for x_file in self.X[index]]
            fnames = [Path(x_file).stem for x_file in self.X[index]]
            # bucketing,
            return wav_batch, label_batch, phoneid_batch, fnames

        else:
            x = self.X[index]
            wav = self._load_wav(x)
            label = torch.LongTensor(
                self.L[self._parse_x_name(x)])
            phones = torch.LongTensor(
                self.Y[self._parse_x_name(x)])
            fnames = x

            return wav, label, phones, fnames

    def collate_fn(self, items):
        if self.bucketing:
            # hack bucketing, return (wavs, labels)
            return items[0][0], items[0][1], items[0][2], items[0][3]
        else:
            return list(zip(*items))
