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
from .model import ConvBank
from .dataset import PronscorDataset
from .train_utils import process_input_forward,  get_phone_weights_as_torch, criterion
import numpy as np


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
        self.phone_weights = get_phone_weights_as_torch(
            self.datarc['phone_weights'])
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
        self.best = defaultdict(lambda: np.inf)

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

        features, labels, phone_ids, lengths = process_input_forward(
            features, labels, phone_ids, self.train_dataset.dataset.class_num)

        phone_weights = self.phone_weights.to(features.device)

        predicted = self.model(features)

        loss = criterion(predicted, labels, weights=phone_weights,
                         norm_per_phone_and_class=self.npc, min_frame_count=0)
        records['loss'] += [loss]

        for pred, lab, l in zip(predicted, labels, lengths):
            records['acc_pos'] += [(pred[:l][lab[:l] == 1]
                                    > 0.666).float().mean()]
            m = (pred[:l][lab[:l] == 0] < 0.333).float().mean()
            if not torch.isnan(m):
                records['acc_neg'] += [m]
            # records['sample_wise_metric'] += [torch.FloatTensor(utter_result).mean().item()]

        return loss

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
        average_loss = torch.FloatTensor(records['loss']).mean().item()
        average_acc_pos = torch.FloatTensor(records['acc_pos']).mean().item()
        average_acc_neg = torch.FloatTensor(records['acc_neg']).mean().item()

        logger.add_scalar(
            f'{prefix}loss',
            average_loss,
            global_step=global_step
        )
        logger.add_scalar(
            f'{prefix}acc_pos',
            average_acc_pos,
            global_step=global_step
        )

        logger.add_scalar(
            f'{prefix}acc_neg',
            average_acc_neg,
            global_step=global_step
        )

        message = f'{prefix}|step:{global_step}|loss:{average_loss}\n'
        save_ckpt = []
        if average_loss < self.best[prefix]:
            self.best[prefix] = average_loss
            message = f'best|{message}'
            name = prefix.split('/')[-1].split('-')[0]
            save_ckpt.append(f'best-states-{name}-step_{global_step}.ckpt')
        with open(self.logging, 'a') as f:
            f.write(message)

        return save_ckpt
