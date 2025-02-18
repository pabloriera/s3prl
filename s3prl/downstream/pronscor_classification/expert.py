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
        self.eval_summarise = self.datarc.get('eval_summarise', 'mean')
        self.class_weight = self.datarc.get('class_weight', False)

        runner = kwargs['runner']
        runner['train_dataloader'] = runner.get('train_dataloader', 'train')
        self.datasets = {}
        if 'phone_path' in self.datarc:
            if 'epa' in self.datarc['phone_path']:
                self.datarc['alignments_path'] = self.datarc['phone_path']+'/alignments'
                self.datarc['splits_path'] = self.datarc['phone_path']+'/splits'
        for split in runner['eval_dataloaders']:
            self.datasets[split] = PronscorDataset(
                split, self.datarc['eval_batch_size'], **self.datarc)
        self.datasets[runner['train_dataloader']] = PronscorDataset(
            runner['train_dataloader'], self.datarc['eval_batch_size'], **self.datarc)
        self.datasets[kwargs['evaluate_split']] = PronscorDataset(
            kwargs['evaluate_split'], self.datarc['eval_batch_size'], **self.datarc)

        self.class_num = self.datasets[runner['train_dataloader']].class_num

        self.objective = nn.BCEWithLogitsLoss()

        self.logging = os.path.join(expdir, 'log.log')
        self.best_loss = defaultdict(lambda: np.inf)
        self.best_f1 = defaultdict(lambda: 0)
        self.best_1mauc = defaultdict(lambda: np.inf)

        # delattr(self, 'model')
        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc[self.modelrc['select']]
        self.model = model_cls(
            self.upstream_dim, output_class_num=self.class_num, **model_conf)

        self.reslayer = None
        if self.reslayer is not None:
            self.reslayer = model_cls(
                self.upstream_dim, output_class_num=self.train_dataset.class_num, **model_conf)
            # self.reslayer.linear.weight.data.fill_(0.0)
            # self.reslayer.linear.bias.data.fill_(0.0)

    """
    Datalaoder Specs:
        Each dataloader should output in the following format:

        [[wav1, wav2, ...], your_other_contents1, your_other_contents2, ...]

        where wav1, wav2 ... are in variable length
        each wav is torch.FloatTensor in cpu with dim()==1 and sample_rate==16000
    """

    # Interface
    def get_dataloader(self, split):
        if 'train' in split:
            batch_size = self.datarc['train_batch_size']
        elif 'dev' in split or 'test' in split:
            batch_size = self.datarc['eval_batch_size']

        dataset = self.datasets[split]

        if self.datarc.get('bucketing', True):
            batch_size = 1

        return DataLoader(
            dataset, batch_size=batch_size,  # for bucketing
            shuffle=True, num_workers=self.datarc['num_workers'],
            drop_last=False, pin_memory=True, collate_fn=dataset.collate_fn
        )

    # Interface

    def forward(self, split, features, labels, phone_ids, filenames, records, **kwargs):
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
            features, labels, phone_ids, self.class_num)

        if self.phone_weights is not None:
            phone_weights = self.phone_weights.to(features.device)
        else:
            phone_weights = None

        predicted = self.model(features)

        if self.reslayer is not None:
            # with torch.no_grad():
            # predicted = self.model(features)
            predicted = predicted + self.reslayer(features)
        # else:

        if 'dev' in split or 'test' in split:
            predicted_copy = torch.clone(predicted)
            labels_copy = torch.clone(labels)

        if self.summarise is not None:
            predicted, labels, _, phones_id_list = get_summarisation(
                phone_ids, labels, predicted, self.summarise)

        # Changes -1, 0, 1 labels to 0, 0.5, 1 for Cross Entropy
        loss = criterion(predicted, (labels+1)/2, class_weight=self.class_weight, weights=phone_weights,
                         norm_per_phone_and_class=self.npc, min_frame_count=0)
        records['loss'] += [loss]

        if 'dev' in split or 'test' in split:
            if self.eval_summarise is None:
                for i in [-1, 1]:
                    _scores = predicted_copy[labels == i]
                    records['phones'] += torch.where(
                        labels_copy == i)[2].tolist()
                    records['scores'] += _scores.tolist()
                    records['labels'] += (i*torch.ones(len(_scores))).tolist()
                    # TODO: check this
                    records['filenames'] += [filenames]

            if self.eval_summarise != self.summarise:
                predicted, labels, _, phones_id_list = get_summarisation(
                    phone_ids, labels_copy, predicted_copy, self.eval_summarise)

            if self.eval_summarise is not None:
                gops_by_phone = torch.sum(predicted, dim=2)
                labels_by_phone = torch.sum(labels, dim=2)
                for i, (phnlist, labs, gops) in enumerate(zip(phones_id_list, labels_by_phone, gops_by_phone)):
                    phrase_labels = labs[:len(phnlist)]
                    phrase_gops = gops[:len(phnlist)]
                    records['phones'] += phnlist.tolist()
                    records['scores'] += phrase_gops.tolist()
                    records['labels'] += phrase_labels.tolist()
                    records['filenames'] += [
                        f'{filenames[i]}_{k}' for k in range(len(phnlist.tolist()))]

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

        # if split[-1].isnumeric():
        #     prefix = f'pronscor/{split[:-1]}'
        # else:
        prefix = f'pronscor/{split}'

        average_loss = torch.FloatTensor(records['loss']).mean().item()
        logger.add_scalar(
            f'{prefix}-loss',
            average_loss,
            global_step=global_step
        )

        message = f'{prefix}|step:{global_step}|loss:{average_loss} \n'
        save_names = []
        if average_loss < self.best_loss[prefix]:
            self.best_loss[prefix] = average_loss
            message = f'best_loss|{message}'
            save_names.append(f'best-loss-{split}.ckpt')
            save_path = Path(self.expdir, f'{split}-best_loss'+'.records')
            torch.save(records, save_path)

        with open(self.logging, 'a') as f:
            f.write(message)

        if 'dev' in split or 'test' in split:

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
                save_path = Path(self.expdir, f'{split}-best_f1'+'.records')
                torch.save(records, save_path)

            average_1mauc = metrics_table.loc['all']['1-AUC']
            logger.add_scalar(
                f'{prefix}-1-AUC',
                average_1mauc,
                global_step=global_step
            )
            message = f'{prefix}|step:{global_step}|1-AUC:{average_1mauc} \n'

            if average_1mauc < self.best_1mauc[prefix]:
                self.best_1mauc[prefix] = average_1mauc
                message = f'best_1mauc|{message}'
                save_names.append(f'best-1-AUC-{split}.ckpt')
                save_path = Path(self.expdir, f'{split}-best_1mauc'+'.records')
                torch.save(records, save_path)

            records_name = f'{split}-{global_step}'
            save_path = Path(self.expdir, records_name+'.records')
            print(f"Saving records to {save_path}")
            torch.save(records, save_path)

            with open(self.logging, 'a') as f:
                f.write(message)

        return save_names
