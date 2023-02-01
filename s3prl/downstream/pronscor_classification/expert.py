###############
# IMPORTATION #
###############
import os
from pathlib import Path
from IPython import embed
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .model import *
from .dataset import PronscorDataset
from .train_utils import process_input_forward,  get_phone_weights_as_torch, criterion, get_summarisation, get_metrics
import numpy as np
import pandas as pd


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
        self.expdir = expdir
        if 'phone_weights' in self.datarc:
            self.phone_weights = get_phone_weights_as_torch(
                self.datarc['phone_weights'])
        else:
            self.phone_weights = None
        self.npc = self.datarc.get('npc', False)
        self.summarise = self.datarc.get('summarise', None)
        self.class_weight = self.datarc.get('class_weight', False)
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
        self.best_loss = defaultdict(lambda: np.inf)
        self.best_f1 = defaultdict(lambda: 0)
        self.best_1mauc = defaultdict(lambda: np.inf)

        # delattr(self, 'model')
        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc[self.modelrc['select']]
        self.model = model_cls(
            self.upstream_dim, output_class_num=self.train_dataset.class_num, **model_conf)

    def _get_train_dataloader(self, dataset):
        if self.datarc['bucketing']:
            batch_size = 1
        else:
            batch_size = self.datarc['train_batch_size']
        return DataLoader(
            dataset, batch_size=batch_size,  # for bucketing
            shuffle=True, num_workers=self.datarc['num_workers'],
            drop_last=False, pin_memory=True, collate_fn=dataset.collate_fn
        )

    def _get_eval_dataloader(self, dataset):
        if self.datarc.get('bucketing', True):
            batch_size = 1
        else:
            batch_size = self.datarc['eval_batch_size']
        return DataLoader(
            dataset, batch_size=batch_size,  # for bucketing
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
            features, labels, phone_ids, self.train_dataset.class_num)

        if self.phone_weights is not None:
            phone_weights = self.phone_weights.to(features.device)
        else:
            phone_weights = None

        predicted = self.model(features)

        if self.summarise is not None:
            predicted, labels, _, phones_id_list = get_summarisation(
                phone_ids, labels, predicted, self.summarise)

        # Changes -1, 0, 1 labels to 0, 0.5, 1 for Cross Entropy
        loss = criterion(predicted, (labels+1)/2, class_weight=self.class_weight, weights=phone_weights,
                         norm_per_phone_and_class=self.npc, min_frame_count=0)
        records['loss'] += [loss]

        if split in ['dev', 'test']:
            if self.summarise is None:
                for i in [-1, 1]:
                    _scores = predicted[labels == i]
                    records['phones'] += torch.where(labels == i)[2].tolist()
                    records['scores'] += _scores.tolist()
                    records['labels'] += (i*torch.ones(len(_scores))).tolist()
            else:
                gops_by_phone = torch.sum(predicted, dim=2)
                labels_by_phone = torch.sum(labels, dim=2)
                for i, (phnlist, labs, gops) in enumerate(zip(phones_id_list, labels_by_phone, gops_by_phone)):
                    phrase_labels = labs[:len(phnlist)]
                    phrase_gops = gops[:len(phnlist)]
                    records['phones'] += phnlist.tolist()
                    records['scores'] += phrase_gops.tolist()
                    records['labels'] += phrase_labels.tolist()

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

        prefix = f'pronscor/{split}'
        average_loss = torch.FloatTensor(records['loss']).mean().item()
        logger.add_scalar(
            f'{prefix}-loss',
            average_loss,
            global_step=global_step
        )

        message = f'{prefix}|step:{global_step}|loss:{average_loss} \n'
        is_best_loss = False
        save_names = []
        if average_loss < self.best_loss[prefix]:
            self.best_loss[prefix] = average_loss
            message = f'best_loss|{message}'
            is_best_loss = True
            save_names.append(f'best-loss-{split}.ckpt')

        with open(self.logging, 'a') as f:
            f.write(message)

        if split in ['dev', 'test']:

            df = pd.DataFrame({k: records[k]
                               for k in ['phones', 'scores', 'labels']})
            df = df[df['phones'] != 0]
            df['labels'] = (df['labels']+1)/2
            metrics_table = get_metrics(df, cost_thrs=None)

            average_f1_scores = metrics_table.loc['all']['F1Score'].max()
            logger.add_scalar(
                f'{prefix}-f1_score',
                average_f1_scores,
                global_step=global_step
            )
            message = f'{prefix}|step:{global_step}|f1:{average_f1_scores} \n'

            if average_f1_scores > self.best_f1[prefix]:
                self.best_f1[prefix] = average_f1_scores
                message = f'best_f1|{message}'
                save_names.append(f'best-f1-{split}.ckpt')

            average_1mauc = metrics_table.loc['all']['1-AUC']
            logger.add_scalar(
                f'{prefix}-1-AUC',
                average_1mauc,
                global_step=global_step
            )
            message = f'{prefix}|step:{global_step}|1-AUC:{average_1mauc} \n'

            if average_1mauc > self.best_1mauc[prefix]:
                self.best_1mauc[prefix] = average_1mauc
                message = f'best_f1|{message}'
                save_names.append(f'best-1-AUC-{split}.ckpt')

            records_name = f'{split}-{global_step}'
            if is_best_loss:
                records_name = f'{records_name}-best_loss'

            save_path = Path(self.expdir, records_name+'.records')
            torch.save(records, save_path)

            with open(self.logging, 'a') as f:
                f.write(message)

        return save_names
