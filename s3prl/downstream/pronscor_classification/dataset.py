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


# TODO: use phone dictionaries not fixed number
from .train_utils import get_phone_dictionaries, NUM_PHONES


HALF_BATCHSIZE_TIME = 2000
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
            phone_path,
            bucket_file,
            sample_rate=16000,
            bucketing=True,
            ** kwargs):

        super(PronscorDataset, self).__init__()

        self.data_root = data_root
        self.phone_path = phone_path
        self.sample_rate = sample_rate
        self.class_num = NUM_PHONES  # NOTE: pre-computed, should not need change
        # Create phone dictionaries
        # self._phone_sym2int_dict, self.phone_int2sym_dict, self.phone_int2node_dict = get_phone_dictionaries(phones_list_path)

        self.Y = {}
        phone_file = open(os.path.join(
            phone_path, 'converted_aligned_phones.txt')).readlines()
        for line in phone_file:
            line = line.strip('\n').split(' ')
            self.Y[line[0]] = [int(p) for p in line[1:]]

        self.L = {}
        label_file = open(os.path.join(
            phone_path, 'converted_aligned_labels.txt')).readlines()
        for line in label_file:
            line = line.strip('\n').split(' ')
            self.L[line[0]] = [int(l) for l in line[1:]]

        if split == 'train':
            train_list = open(os.path.join(
                phone_path, 'train_split.txt')).readlines()
            usage_list = [line.strip('\n') for line in train_list]
        elif split == 'dev':
            dev_list = open(os.path.join(
                phone_path, 'dev_split.txt')).readlines()
            usage_list = [line.strip('\n') for line in dev_list]
        elif split == 'test':
            test_list = open(os.path.join(
                phone_path, 'test_split.txt')).readlines()
            usage_list = [line.strip('\n') for line in test_list]
            # no separé un dev. ver si me lo cobra.
            # if split == 'dev':
            #    usage_list = [line for line in usage_list if not line.split(
            #        '-')[1].lower() in TEST_SPEAKERS]  # held-out speakers from test
            # else:
            #    usage_list = [line for line in usage_list if line.split(
            #        '-')[1].lower() in TEST_SPEAKERS]  # 24 core test speakers, 192 sentences, 0.16 hr
        else:
            raise ValueError(
                'Invalid \'split\' argument for dataset: PronscorDataset!')
        usage_list = {line.strip('\n'): None for line in usage_list}
        print('[Dataset] - # phone classes: ' + str(self.class_num) +
              ', number of data for ' + split + ': ' + str(len(usage_list)))
        # Read table for bucketing
        assert os.path.isdir(
            bucket_file), f'Missing {bucket_file} Please first run `preprocess/generate_len_for_bucket.py` to get bucket file.'
        if split == 'train':
            table = pd.read_csv(os.path.join(bucket_file, 'TRAIN16k.csv')).sort_values(
                by=['length'], ascending=False)
        elif split == 'dev':
            table = pd.read_csv(os.path.join(bucket_file, 'DEV16k.csv')).sort_values(
                by=['length'], ascending=False)
        elif split == 'test':
            table = pd.read_csv(os.path.join(bucket_file, 'TEST16k.csv')).sort_values(
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
                if self._parse_x_name(x) in usage_list:
                    self.X.append(batch_x)
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
            # bucketing,
            return wav_batch, label_batch, phoneid_batch

        else:
            x = self.X[index]
            wav = self._load_wav(x)
            label = torch.LongTensor(
                self.L[self._parse_x_name(x)])
            phones = torch.LongTensor(
                self.Y[self._parse_x_name(x)])
            return wav, label, phones

    def collate_fn(self, items):
        if self.bucketing:
            # hack bucketing, return (wavs, labels)
            return items[0][0], items[0][1], items[0][2]
        else:
            return list(zip(*items))
