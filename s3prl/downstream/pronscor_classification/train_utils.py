
import torch
import numpy as np
import yaml
from IPython import embed

def format_labels(labels_array, phones_array):
    '''
    Function that receives two flat tensors of labels and phone ids
    and turns them into a quasi one hot encoded (x_X) 2d tensor with
    labels in the frames of each target phone in the phrase
    '''
    idx = np.where(phones_array[:-1] != phones_array[1:])[0]
    labels_2darray = np.zeros((len(phones_array),40))
    start = 0
    for elem in idx:
        tmp_2darray = np.zeros((len(phones_array),40))
        end = elem
        col = phones_array[end]
        lab = labels_array[end]*2-1
        tmp_2darray[col][start:end+1] = lab
        labels_2darray += tmp_2darray
        start = end+1
    return(labels_2darray)


def cum_matrix(phones_array):
    index = np.where(phones_array[:-1] != phones_array[1:])[0]
    rows = []
    frame_counts = []
    start = 0
    for i in index:
        tmp_row = np.zeros(len(phones))
        end = i
        tmp_row[start:end+1] = 1
        num_frames = np.sum(tmp_row)
        rows.append(tmp_row)
        frame_counts.append(num_frames)
        start = end+1
    res = np.stack(rows, axis=0)
    return(res, frame_counts)



#phone_sym2int_dict:  Dictionary mapping phone symbol to integer given a phone list path
#phone_int2sym_dict:  Dictionary mapping phone integer to symbol given a phone list path
#phone_int2node_dict: Dictionary mapping phone symbol to the index of the node in the networks's output layer
#NOTE: The node number in the output layer is not the same as the phone number, as some phones will not be scored

def get_phone_dictionaries(phone_list_path):
    #Open file that contains list of pure phones
    phones_list_fh = open(phone_list_path, "r")

    phone_sym2int_dict  = {}
    phone_int2sym_dict  = {}
    phone_int2node_dict = {}
    current_node_index  = 0
    #Populate the dictionaries
    for line in phones_list_fh.readlines():
        line = line.split()
        phone_symbol = line[0]
        phone_number = int(line[1])
        use_phone    = bool(int(line[2]))
        if use_phone:
            phone_sym2int_dict[phone_symbol]    = phone_number
            phone_int2sym_dict[phone_number]    = phone_symbol
            phone_int2node_dict[phone_number]   = current_node_index
            current_node_index += 1

    return phone_sym2int_dict, phone_int2sym_dict, phone_int2node_dict


def get_phone_weights_as_torch(phone_weights_path):
    phone_weights_fh = open(phone_weights_path)
    phone_weights = yaml.safe_load(phone_weights_fh)
    weights_list = []
    for phone, weight in phone_weights.items():
        weights_list.append(weight)
    phone_weights = weights_list
    return torch.tensor(phone_weights)




def calculate_loss(outputs, mask, labels, phone_weights=None, norm_per_phone_and_class=False, min_frame_count=0):
    weights = mask *1

    if phone_weights is not None:
        weights = weights * phone_weights

    if norm_per_phone_and_class:
        frame_count = torch.sum(mask, dim=[0,1])
        weights = weights * torch.nan_to_num(1 / frame_count)
        if min_frame_count > 0:
            # Set to 0 the weights for the phones with too few cases in this batch
            weights[:,:,frame_count<min_frame_count] = 0.0

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none', weight=weights)
    
    return loss_fn(outputs, labels), torch.sum(weights)



def criterion(batch_outputs, batch_labels, weights=None, norm_per_phone_and_class=False, min_frame_count=0):

    batch_labels_for_loss = (batch_labels+1)/2

    loss_pos, sum_weights_pos = calculate_loss(batch_outputs, batch_labels ==  1, batch_labels_for_loss, 
        phone_weights=weights, norm_per_phone_and_class=norm_per_phone_and_class, min_frame_count=min_frame_count)
    
    loss_neg, sum_weights_neg = calculate_loss(batch_outputs, batch_labels == -1, batch_labels_for_loss, 
        phone_weights=weights, norm_per_phone_and_class=norm_per_phone_and_class, min_frame_count=min_frame_count)

    
    total_loss = (loss_pos + loss_neg).sum()

    if not norm_per_phone_and_class:
        total_weights = sum_weights_pos + sum_weights_neg
        total_loss /= total_weights

    
    return total_loss


