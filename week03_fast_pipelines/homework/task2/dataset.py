from typing import Optional

import torch
import os
import random
from collections import defaultdict
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler, IterableDataset
from transformers import AutoTokenizer
from tqdm import tqdm


MAX_LENGTH = 640


class BrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_len = max_length
        
        with open(os.path.join(data_path, "train-00000-of-00002.txt"), encoding="utf-8") as file:
            list_of_strings = list(file)
        self.tensors_list = []
        for string in tqdm(list_of_strings[:10000]):
            unpadded_tensor = self.tokenizer(string, max_length=self.max_len, return_tensors="pt")['input_ids'][0]
            padding_len = max(self.max_len - len(unpadded_tensor), 0)
            self.tensors_list.append(torch.cat([unpadded_tensor, torch.full((padding_len,), 0)], dim=0))
            
    def __len__(self):
        return len(self.tensors_list)
            
    def __getitem__(self, idx: int):
        return self.tensors_list[idx][:-1], self.tensors_list[idx][1:]

class BigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_len = max_length
        
        with open(os.path.join(data_path, "train-00000-of-00002.txt"), encoding="utf-8") as file:
            list_of_strings = list(file)
            
        self.tensors_list = [self.tokenizer(string, max_length=self.max_len, return_tensors="pt")['input_ids'][0] for string in list_of_strings[:10000]]      
        
    def __len__(self):
        return len(self.tensors_list)

    def __getitem__(self, idx: int):
        return self.tensors_list[idx]


class UltraBigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH, max_diff: int = 0):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_len = max_length + 1
        
        with open(os.path.join(data_path, "train-00000-of-00002.txt"), encoding="utf-8") as file:
            list_of_strings = list(file)
        self.tensors_list = []
        self.bins_and_ids = defaultdict(list)
        for idx, string in enumerate(list_of_strings[:10000]):
            unpadded_tensor = self.tokenizer(string, max_length=self.max_len, return_tensors="pt")['input_ids'][0]
            self.tensors_list.append(unpadded_tensor)
            self.bins_and_ids[len(unpadded_tensor) // max_diff].append(idx)
    
    def __len__(self):
        return len(self.tensors_list)

    def __getitem__(self, idx: int):
        return self.tensors_list[idx]


class UltraDuperBigBrainDataset(IterableDataset):
    def __init__(self, data_path: str, sample_len=1024, max_length: int = MAX_LENGTH):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_len = max_length
        self.sample_len = sample_len
        with open(os.path.join(data_path, "train-00000-of-00002.txt"), encoding="utf-8") as file:
            list_of_strings = list(file)
            
        self.tensors_list = [self.tokenizer(string, max_length=self.max_len, return_tensors="pt")['input_ids'][0] for string in list_of_strings[:10000]]
        
        

    def __iter__(self):
        random.shuffle(self.tensors_list)
        curr_id = 0
        finished_flag = False
        while True:
            curr_remainder = self.sample_len
            curr_sample_ids = []
            lens_array = []
            while True:
                curr_tensor_len = len(self.tensors_list[curr_id])
                if curr_remainder - (curr_tensor_len - 1) > 0:
                    curr_sample_ids.append(curr_id)
                    curr_remainder -= (curr_tensor_len - 1)
                    lens_array.append(curr_tensor_len - 1)
                    curr_id += 1
                    if curr_id == len(self.tensors_list):
                        curr_id = 0
                        finished_flag = True
                else:
                    sample_x = torch.cat([self.tensors_list[idx][:-1] for idx in curr_sample_ids] + [self.tensors_list[curr_id][:curr_remainder]], dim=0)
                    sample_y = torch.cat([self.tensors_list[idx][1:] for idx in curr_sample_ids] + [self.tensors_list[curr_id][1:curr_remainder+1]], dim=0)
                    break
                    
            attn_mask = create_supa_hot_fire_attention_mask(self.sample_len, lens_array + [curr_remainder]) * float("-inf")      
            yield sample_x, sample_y, attn_mask.expand(8, self.sample_len, self.sample_len)
            
            curr_id += 1
            if curr_id == len(self.tensors_list):
                curr_id = 0
                finished_flag = True
                
            if finished_flag:
                break
    
    
def create_supa_hot_fire_attention_mask(general_mask_size, list_of_blocks_sizes, device='cuda'):
    inverted_diagonal_blocks =[1 - torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(device) for seq_len in list_of_blocks_sizes]
    mask_with_ones = (1 - torch.block_diag(*inverted_diagonal_blocks))[:general_mask_size, :general_mask_size]
    return torch.where(mask_with_ones == 1, float('-inf'), 0).to(device)

    

def collate_fn(batch: list[tuple[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad each sequence of the incoming sequences list
    :param batch: a list of the objects received from the dataset by __getitem__
    :param max_length: maximum sequence length to pad to (for "Brain" approach only) ### Big Brain approach mb???
    :return: tuple of padded sequences and corresponding training targets
    """
    padding_value = 0
    max_len = max(len(seq) for seq in batch)
    padded_sequences = [torch.cat([seq, torch.full((max_len - len(seq),), padding_value)], dim=0) for seq in batch]      
    padded_tensor = torch.stack(padded_sequences)
    return padded_tensor[:, :-1], padded_tensor[:, 1:]


class UltraBigBrainBatchSampler(Sampler):

    def __init__(self, batch_size: int, bins_and_ids, max_length: Optional[int] = MAX_LENGTH):
        self.bins_and_ids = bins_and_ids
        self.bin_id_and_bounds_list = []
        for bin_id, idx_list in bins_and_ids.items():
            self.bin_id_and_bounds_list += [(bin_id, left_bound, min(left_bound + batch_size, len(idx_list))) for left_bound in range(0, len(idx_list), batch_size)]

    def __len__(self):
        return len(self.bin_id_and_bounds_list)

    def __iter__(self):
        visited_bins = set()
        random.shuffle(self.bin_id_and_bounds_list)
        for bin_id, l_b, r_b in self.bin_id_and_bounds_list:
            if bin_id not in visited_bins:
                visited_bins.add(bin_id)
                random.shuffle(self.bins_and_ids[bin_id])
            yield self.bins_and_ids[bin_id][l_b:r_b]
        
