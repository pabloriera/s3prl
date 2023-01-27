
import torch
import numpy as np
import yaml
from IPython import embed
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
NUM_PHONES = 40


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
            res[j, phone]=min_value
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

    if summarise == 'lpp':
        logits = torch.matmul(summarisation_mask, masked_outputs)
        logits = torch.div(logits, frame_counts)
        
    elif summarise == 'softmin':
        one_minus_outputs = 1-masked_outputs
        M = torch.exp(one_minus_outputs)
        M_masked = M*mask
        N = torch.matmul(summarisation_mask, M_masked)
        N_vec = torch.sum(N, dim=2)
        xpnd_N_vec = N_vec.unsqueeze(2).repeat(1,1,summarisation_mask.shape[2])
        cumm_N = summarisation_mask*xpnd_N_vec
        cumm_N_vec = torch.sum(cumm_N, dim=1)
        xpnd_cumm_N_vec = cumm_N_vec.unsqueeze(2).repeat(1,1,40)
        xpnd_cumm_N_vec[xpnd_cumm_N_vec==0]=1
        S = torch.div(M_masked,xpnd_cumm_N_vec)
        masked_outputs = (S*masked_outputs)
        logits = torch.matmul(summarisation_mask, masked_outputs)
    
    else:
        min_phrase_list, n_phones = get_min(phone_ids, predicted)
        aux_list = []
        for phrase in min_phrase_list:
            aux = torch.zeros(max_phone_count, n_phones) 
            aux[:phrase.shape[0], :] = phrase
            aux_list.append(aux)
        logits = torch.stack(aux_list, dim=0).to(predicted.device)


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

        if class_weight:
            total_loss = (loss_pos.sum()/sum_weights_pos + loss_neg.sum()/sum_weights_pos)
        else:
            total_loss = (loss_pos + loss_neg).sum()        
            total_weights = sum_weights_pos + sum_weights_pos
            total_loss /= total_weights

    return total_loss
