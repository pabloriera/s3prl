
from matplotlib import gridspec
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import yaml
from IPython import embed
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support
from scipy import interpolate
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

NUM_PHONES = 40


def get_metrics(df, cost_thrs=None, f1_thr=None):

    metrics = dict()

    metrics['all'] = compute_metrics(df, cost_thr=None, f1_thr=f1_thr)

    for phone, g in df.groupby('phones'):
        cost_thr = cost_thrs[phone]['MinCostThr'] if cost_thrs is not None else None
        metrics[phone] = compute_metrics(g, cost_thr=cost_thr, f1_thr=f1_thr)

    metrics_table = pd.DataFrame(metrics).T
    metrics_table.index.name = 'phones'

    return metrics_table


def compute_metrics(df, cost_fp=0.5, cost_thr=None, f1_thr=None, pos_label=0):
    if pos_label == 0:
        scores = -df['scores']
        labels = 1-df['labels']
    elif pos_label == 1:
        scores = df['scores']
        labels = df['labels']

    if f1_thr is None:
        precision, recall, f1_thr = precision_recall_curve(
            labels, scores)

        numerator = 2 * recall * precision
        denom = recall + precision
        f1_scores = np.divide(
            numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
    else:
        precision, recall, f1_scores, _ = precision_recall_fscore_support(
            labels, scores > f1_thr, average='binary')

        # TR = ((df['scores'] < 0) & (df['label'] == 0)).sum()
        # FR = ((df['scores'] < 0) & (df['label'] == 1)).sum()
        # FA = ((df['scores'] > 0) & (df['label'] == 0)).sum()

        # precision = TR/(TR+FR)
        # recall = TR/(TR+FA)
        # f1_scores = 2*(precision*recall)/(precision+recall)

    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1-tpr

    # Use the best (cheating) threshold to get the min_cost
    cost_normalizer = min(cost_fp, 1.0)
    cost = (cost_fp * fpr + fnr)/cost_normalizer
    min_cost_idx = np.argmin(cost)
    min_cost_thr = thr[min_cost_idx]
    min_cost = cost[min_cost_idx]
    min_cost_fpr = fpr[min_cost_idx]
    min_cost_fnr = fnr[min_cost_idx]

    if cost_thr is not None:
        det_pos = labels[scores > cost_thr]
        det_neg = labels[scores <= cost_thr]
        act_cost_fpr = np.sum(det_pos == 0)/np.sum(labels == 0)
        act_cost_fnr = np.sum(det_neg == 1)/np.sum(labels == 1)
        act_cost = (cost_fp * act_cost_fpr + act_cost_fnr)/cost_normalizer
#        print(min_cost, act_cost, cost_thr, min_cost_thr)
    else:
        act_cost_fpr = min_cost_fpr
        act_cost_fnr = min_cost_fnr
        act_cost = min_cost

    aucv = auc(fpr, tpr)
    eerv = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)

    metrics = {
        "1-AUC": 1-aucv,
        "EER": eerv,
        "MinCost": min_cost,
        "MinCostThr": min_cost_thr,
        "FPR4MinCost": min_cost_fpr,
        "FNR4MinCost": min_cost_fnr,
        "ActCost": act_cost,
        "FPR4ActCost": act_cost_fpr,
        "FNR4ActCost": act_cost_fnr,
        "POS_COUNT": np.sum(labels),
        "NEG_COUNT": len(labels)-np.sum(labels),
        "FPR": fpr,
        "FNR": fnr,
        "POS": scores[labels == 1],
        "NEG": scores[labels == 0],
        "Recall": recall,
        "Precision": precision,
        "F1Score": f1_scores,
        "F1Thr": f1_thr
    }

    return metrics


def tile_representations(reps, factor):
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


def match_length(inputs, labels):
    """
    Since the upstream extraction process can sometimes cause a mismatch
    between the seq lenth of inputs and labels:
    - if len(inputs) > len(labels), we truncate the final few timestamp of inputs to match the length of labels
    - if len(inputs) < len(labels), we duplicate the last timestep of inputs to match the length of labels
    Note that the length of labels should never be changed.
    """
    input_len, label_len = inputs.size(1), labels.size(-1)

    factor = int(round(label_len / input_len))
    if factor > 1:
        inputs = tile_representations(inputs, factor)
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


def process_input_forward(features, labels, phone_ids, num_phones, silence_id=0):
    """
    TODO: Describeme
    """
    lengths = torch.LongTensor([len(l) for l in labels])

    features = pad_sequence(features, batch_first=True)
    phone_ids = pad_sequence(
        phone_ids, batch_first=True, padding_value=silence_id)
    labels = pad_sequence(labels, batch_first=True,
                          padding_value=0).to(features.device)
    features, labels = match_length(features, labels)

    labels2d_list = []
    for lab, phn in zip(labels, phone_ids):
        labels_2darray = format_labels(lab, phn, num_phones)
        labels2d_list.append(labels_2darray)

    labels = torch.stack(labels2d_list)

    return features, labels, phone_ids, lengths


def format_labels(labels_array, phones_array, num_phones):
    '''
    Function that receives two flat tensors of labels and phone ids
    and turns them into a quasi one hot encoded (x_X) 2d tensor with
    labels in the frames of each target phone in the phrase
    '''

    x = torch.zeros((len(phones_array), num_phones),
                    dtype=torch.float).to(labels_array.device)
    x[torch.arange(len(phones_array)), phones_array
      ] = labels_array.float()

    return x


def get_phones_masks(phone_list):
    n_frames = len(phone_list)
    index = torch.where(phone_list[:-1] != phone_list[1:])[0]
    index = torch.hstack((index, torch.tensor(n_frames-1)))
    start = 0
    res = torch.zeros((len(index), n_frames))
    for i, end in enumerate(index):
        res[i, start:end+1] = 1
        start = end+1
    return res, phone_list[index]


def get_max(phone_ids, predicted):
    n_phones = predicted.shape[2]
    max_batch_list = []
    for i, phone_list in enumerate(phone_ids):
        n_frames = len(phone_list)
        index = torch.where(phone_list[:-1] != phone_list[1:])[0]
        index = torch.hstack((index, torch.tensor(n_frames-1)))
        start = 0
        res = torch.zeros((len(index), n_phones)).to(predicted.device)
        for j, end in enumerate(index):
            phone = phone_list[end]
            max_value = torch.max(predicted[i][start:end+1, phone])
            res[j, phone] = max_value
            start = end+1
        max_batch_list.append(res)
    return max_batch_list, n_phones


def get_min(phone_ids, predicted):
    n_phones = predicted.shape[2]
    min_batch_list = []
    for i, phone_list in enumerate(phone_ids):
        n_frames = len(phone_list)
        index = torch.where(phone_list[:-1] != phone_list[1:])[0]
        index = torch.hstack((index, torch.tensor(n_frames-1)))
        start = 0
        res = torch.zeros((len(index), n_phones)).to(predicted.device)
        for j, end in enumerate(index):
            phone = phone_list[end]
            min_value = torch.min(predicted[i][start:end+1, phone])
            res[j, phone] = min_value
            start = end+1
        min_batch_list.append(res)
    return min_batch_list, n_phones


def get_summarisation(phone_ids, labels, predicted, summarise):

    mask = abs(labels)
    masked_outputs = predicted*mask

    phone_masks_list = []
    phones_id_list = []

    for phone_list in phone_ids:
        phone_mask, phonesids = get_phones_masks(phone_list)
        phone_masks_list.append(phone_mask)
        phones_id_list.append(phonesids)

    max_phone_count = max(len(x) for x in phones_id_list)
    n_frames = labels.shape[1]

    summarisation_mask = torch.zeros(
        (labels.shape[0], max_phone_count, n_frames), device=predicted.device)

    for i, phone_mask in enumerate(phone_masks_list):
        summarisation_mask[i, :phone_mask.shape[0], :] = phone_mask

    labels = torch.sign(torch.matmul(summarisation_mask, labels))
    frame_counts = torch.matmul(summarisation_mask, mask)
    frame_counts[frame_counts == 0] = 1

    if summarise == 'softmin':
        one_minus_outputs = 1-masked_outputs
        M = torch.exp(one_minus_outputs)
        M_masked = M*mask
        N = torch.matmul(summarisation_mask, M_masked)
        N_vec = torch.sum(N, dim=2)
        xpnd_N_vec = N_vec.unsqueeze(2).repeat(
            1, 1, summarisation_mask.shape[2])
        cumm_N = summarisation_mask*xpnd_N_vec
        cumm_N_vec = torch.sum(cumm_N, dim=1)
        xpnd_cumm_N_vec = cumm_N_vec.unsqueeze(2).repeat(1, 1, 40)
        xpnd_cumm_N_vec[xpnd_cumm_N_vec == 0] = 1
        S = torch.div(M_masked, xpnd_cumm_N_vec)
        masked_outputs = (S*masked_outputs)
        logits = torch.matmul(summarisation_mask, masked_outputs)
    elif summarise == 'min':
        min_phrase_list, n_phones = get_min(phone_ids, predicted)
        aux_list = []
        for phrase in min_phrase_list:
            aux = torch.zeros(max_phone_count, n_phones)
            aux[:phrase.shape[0], :] = phrase
            aux_list.append(aux)
        logits = torch.stack(aux_list, dim=0).to(predicted.device)
    elif summarise == 'max':
        min_phrase_list, n_phones = get_max(phone_ids, predicted)
        logits = torch.zeros(len(min_phrase_list),
                             max_phone_count, n_phones, device=predicted.device)
        for i, phrase in enumerate(min_phrase_list):
            logits[i, :phrase.shape[0], :] = phrase
    elif summarise == 'lpp' or summarise == 'mean':
        logits = torch.matmul(summarisation_mask, masked_outputs)
        logits = torch.div(logits, frame_counts)
    else:
        raise Exception('Wrong summarise method')

    return logits, labels, frame_counts, phones_id_list


# phone_sym2int_dict:  Dictionary mapping phone symbol to integer given a phone list path
# phone_int2sym_dict:  Dictionary mapping phone integer to symbol given a phone list path
# phone_int2node_dict: Dictionary mapping phone symbol to the index of the node in the networks's output layer
# NOTE: The node number in the output layer is not the same as the phone number, as some phones will not be scored

def get_phone_dictionaries(phone_list_path):
    # Open file that contains list of pure phones
    phones_list_fh = open(phone_list_path, "r")

    phone_sym2int_dict = {}
    phone_int2sym_dict = {}
    phone_int2node_dict = {}
    current_node_index = 0
    # Populate the dictionaries
    for line in phones_list_fh.readlines():
        line = line.split()
        phone_symbol = line[0]
        phone_number = int(line[1])
        use_phone = bool(int(line[2]))
        if use_phone:
            phone_sym2int_dict[phone_symbol] = phone_number
            phone_int2sym_dict[phone_number] = phone_symbol
            phone_int2node_dict[phone_number] = current_node_index
            current_node_index += 1

    return phone_sym2int_dict, phone_int2sym_dict, phone_int2node_dict


def get_phone_weights_as_torch(phone_weights_path):
    with open(phone_weights_path, 'r') as fp:
        phone_weights = yaml.safe_load(fp)
    weights_list = []
    for phone, weight in phone_weights.items():
        weights_list.append(weight)
    phone_weights = weights_list
    return torch.tensor(phone_weights)


def calculate_loss(outputs, mask, labels, phone_weights=None, norm_per_phone_and_class=False, min_frame_count=0):
    weights = mask * 1

    if phone_weights is not None:
        weights = weights * phone_weights

    if norm_per_phone_and_class:
        # batchnorm
        frame_count = torch.sum(mask, dim=[0, 1])
        weights = weights * torch.nan_to_num(1 / frame_count)
        if min_frame_count > 0:
            # Set to 0 the weights for the phones with too few cases in this batch
            weights[:, :, frame_count < min_frame_count] = 0.0
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none', weight=weights)

    return loss_fn(outputs, labels), torch.sum(weights)


def criterion(batch_outputs, batch_labels, class_weight=False, weights=None, norm_per_phone_and_class=False,  min_frame_count=0):

    if not class_weight and weights is None and not norm_per_phone_and_class:
        mask = batch_labels != 0.5
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none', weight=mask)
        total_loss = loss_fn(batch_outputs, batch_labels).sum()/mask.sum()
    else:
        loss_pos, sum_weights_pos = calculate_loss(batch_outputs, batch_labels == 1, batch_labels,
                                                   phone_weights=weights, norm_per_phone_and_class=norm_per_phone_and_class, min_frame_count=min_frame_count)

        loss_neg, sum_weights_neg = calculate_loss(batch_outputs, batch_labels == 0, batch_labels,
                                                   phone_weights=weights, norm_per_phone_and_class=norm_per_phone_and_class, min_frame_count=min_frame_count)

        if isinstance(class_weight, bool):
            if class_weight:
                total_loss = (loss_pos.sum()/sum_weights_pos +
                              loss_neg.sum()/sum_weights_neg)
        elif isinstance(class_weight, str):
            if class_weight == 'only_neg':
                total_loss = loss_neg.sum()/sum_weights_neg
            elif class_weight == 'only_pos':
                total_loss = loss_pos.sum()/sum_weights_pos
        else:
            total_loss = (loss_pos + loss_neg).sum() / \
                (sum_weights_pos + sum_weights_neg)

    return total_loss


class DynamicSubplots():
    def __init__(self, figsize=None, ncol=3):
        self.ncol = ncol
        self.count = 0
        self.row = 1
        self.col = 0
        self.fig = plt.figure(figsize=figsize)
        self.figsize = self.fig.get_size_inches()

    def __call__(self):
        self.count += 1
        if self.count <= self.ncol:
            self.col += 1
        if self.count % self.ncol == 1:
            self.row += 1
        self.fig.set_size_inches(
            self.figsize[0]*self.col, self.figsize[1]*self.row)
        gs = gridspec.GridSpec(self.row, self.col)
        for i, ax in enumerate(self.fig.axes):
            ax.set_position(gs[i].get_position(self.fig))
            ax.set_subplotspec(gs[i])
        new_ax = self.fig.add_subplot(self.row, self.col, self.count)
        plt.sca(new_ax)
        return self
