###############
# IMPORTATION #
###############
import os
import math
import random
from IPython import embed
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from .model import ConvBank
from .dataset import PronscorDataset
from .train_utils import format_labels #, criterion 
import numpy as np

from s3prl.downstream.pronscor_classification.train_utils import get_phone_weights_as_torch, criterion



class DownstreamExpert(nn.Module):
    """
    Adapted from phone linear expert

    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        self.phone_weights = get_phone_weights_as_torch(self.datarc['phone_weights'])
        self.npc = self.datarc['npc']

        self.train_dataset = PronscorDataset(
            'train', self.datarc['train_batch_size'], **self.datarc)
        self.dev_dataset = PronscorDataset(
            'dev', self.datarc['eval_batch_size'], **self.datarc)
        self.test_dataset = PronscorDataset(
            'test', self.datarc['eval_batch_size'], **self.datarc)
        # self.model = Model(input_dim=self.upstream_dim,
        #    output_class_num=self.train_dataset.class_num, **self.modelrc)
        self.objective = nn.BCEWithLogitsLoss()

        self.logging = os.path.join(expdir, 'log.log')
        self.best = defaultdict(lambda: 0)

        # delattr(self, 'model')
        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc[self.modelrc['select']]
        self.model = model_cls(
            self.upstream_dim, output_class_num=self.train_dataset.class_num, **model_conf)

    def _get_train_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=1,  # for bucketing
            shuffle=True, num_workers=self.datarc['num_workers'],
            drop_last=False, pin_memory=True, collate_fn=dataset.collate_fn
        )

    def _get_eval_dataloader(self, dataset):

        return DataLoader(
            dataset, batch_size=1,  # for bucketing
            shuffle=False, num_workers=self.datarc['num_workers'],
            drop_last=False, pin_memory=True, collate_fn=dataset.collate_fn
        )

    """
    Datalaoder Specs:
        Each dataloader should output in the following format:

        [[wav1, wav2, ...], your_other_contents1, your_other_contents2, ...]

        where wav1, wav2 ... are in variable length
        each wav is torch.FloatTensor in cpu with dim()==1 and sample_rate==16000
    """

    def get_train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset)

    def get_dev_dataloader(self):
        return self._get_eval_dataloader(self.dev_dataset)

    def get_test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)

    # Interface
    def get_dataloader(self, split):
        return eval(f'self.get_{split}_dataloader')()
    

    def _tile_representations(self, reps, factor):
        """ 
        Tile up the representations by `factor`.
        Input - sequence of representations, shape: (batch_size, seq_len, feature_dim)
        Output - sequence of tiled representations, shape: (batch_size, seq_len * factor, feature_dim)
        """
        assert len(reps.shape) == 3, 'Input argument `reps` has invalid shape: {}'.format(
            reps.shape)
        tiled_reps = reps.repeat(1, 1, factor)
        tiled_reps = tiled_reps.reshape(
            reps.size(0), reps.size(1)*factor, reps.size(2))
        return tiled_reps

    def _match_length(self, inputs, labels):
        """
        Since the upstream extraction process can sometimes cause a mismatch
        between the seq lenth of inputs and labels:
        - if len(inputs) > len(labels), we truncate the final few timestamp of inputs to match the length of labels
        - if len(inputs) < len(labels), we duplicate the last timestep of inputs to match the length of labels
        Note that the length of labels should never be changed.
        """
        input_len, label_len = inputs.size(1), labels.size(-1)

        factor = int(round(label_len / input_len))
        print(factor)
        if factor > 1:
            inputs = self._tile_representations(inputs, factor)
            input_len = inputs.size(1)

        if input_len > label_len:
            inputs = inputs[:, :label_len, :]
        elif input_len < label_len:
            # (batch_size, 1, feature_dim)
            pad_vec = inputs[:, -1, :].unsqueeze(1)
            # (batch_size, seq_len, feature_dim), where seq_len == labels.size(-1)
            inputs = torch.cat(
                (inputs, pad_vec.repeat(1, label_len-input_len, 1)), dim=1)
        return inputs, labels


    # Interface
    def forward(self, split, features, labels, phone_ids, records, **kwargs):
        """
        Args:
            features:
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args

            labels:
                the frame-wise phone labels

            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records every log_step

            logger:
                Tensorboard SummaryWriter, given here for logging/debugging convenience
                please use f'{prefix}your_content_name' as key name
                to log your customized contents

            prefix:
                used to indicate downstream and train/test on Tensorboard
                eg. 'phone/train-'

            global_step:
                global_step in runner, which is helpful for Tensorboard logging

        Return:
            loss:
                the loss to be optimized, should not be detached
        """
        lengths = torch.LongTensor([len(l) for l in labels])

        features = pad_sequence(features, batch_first=True)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100).to(features.device)
        features, labels = self._match_length(features, labels)
        
        labels = labels.detach().cpu().numpy()
        phone_ids = pad_sequence(phone_ids, batch_first=True, padding_value=-100).detach().cpu().numpy()
        predicted = self.model(features)
        
        labels2d_list = []
        for lab, phn in zip(labels, phone_ids):
            labels_2darray = format_labels(lab, phn)
            labels2d_list.append(labels_2darray)

        labels2tensor = labels2d_list[0].T
        for l in labels2d_list[1:]:
            labels2tensor = np.dstack((labels2tensor,l.T))
        labels = torch.from_numpy(labels2tensor.T).to(features.device)
        phone_weights = self.phone_weights.to(features.device)
        loss = criterion(predicted, labels, weights=phone_weights, norm_per_phone_and_class=self.npc, min_frame_count=0)

        return loss, predicted, labels
        

    def log_records(self, split, records, logger, global_step, **kwargs):
        """
        Args:
            records:
                defaultdict(list), contents already appended

            logger:
                Tensorboard SummaryWriter
                please use f'{prefix}your_content_name' as key name
                to log your customized contents

            prefix:
                used to indicate downstream and train/test on Tensorboard
                eg. 'phone/train-'

            global_step:
                global_step in runner, which is helpful for Tensorboard logging
        """
        prefix = f'pronscor/{split}-'
        average = torch.FloatTensor(records['acc']).mean().item()

        logger.add_scalar(
            f'{prefix}acc',
            average,
            global_step=global_step
        )
        message = f'{prefix}|step:{global_step}|acc:{average}\n'
        save_ckpt = []
        if average > self.best[prefix]:
            self.best[prefix] = average
            message = f'best|{message}'
            name = prefix.split('/')[-1].split('-')[0]
            save_ckpt.append(f'best-states-{name}.ckpt')
        with open(self.logging, 'a') as f:
            f.write(message)

        return save_ckpt
