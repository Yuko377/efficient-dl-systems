from enum import Enum

import torch
import numpy as np
from gpt2like import GPT2LikeModel
from dataset import *
from torch.utils.data import DataLoader

class DataMode(Enum):
    BRAIN = 1
    BIG_BRAIN = 2
    ULTRA_BIG_BRAIN = 3
    ULTRA_DUPER_BIG_BRAIN = 4


def get_gpt2_model() -> torch.nn.Module:
    pass


def run_epoch(data_mode: DataMode) -> None:
    pass

BATCH_SIZE = 16
UBB_MAX_DIFF = 640
UDBB_SAMPLE_LEN = 650

def measure_epoch_time(data_mode, logs):
    assert torch.cuda.is_available()
    device = 'cuda'
    
    if data_mode == DataMode.BRAIN:
        dataset = BrainDataset('./wikitext-103-raw-v1')
        loader = DataLoader(dataset, batch_size=BATCH_SIZE)
        it = iter(loader)
    elif data_mode == DataMode.BIG_BRAIN:
        dataset = BigBrainDataset('./wikitext-103-raw-v1')
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
        it = iter(loader)
    elif data_mode == DataMode.ULTRA_BIG_BRAIN:
        dataset = UltraBigBrainDataset('./wikitext-103-raw-v1', max_diff = UBB_MAX_DIFF)
        ubb_sampler = UltraBigBrainBatchSampler(BATCH_SIZE, dataset.bins_and_ids)
        loader = DataLoader(dataset, collate_fn=collate_fn, batch_sampler=ubb_sampler)
        it = iter(loader)
    elif data_mode == DataMode.ULTRA_DUPER_BIG_BRAIN:
        dataset = UltraDuperBigBrainDataset('./wikitext-103-raw-v1', sample_len=UDBB_SAMPLE_LEN)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE)
        it = iter(loader)
    else:
        raise ValueError
        

    dataset_name = dataset.__class__.__name__
    logs[dataset_name] = {}
    batch_processing_time = []
    
    model = GPT2LikeModel().to(device)

    with torch.no_grad():
        if data_mode == DataMode.ULTRA_DUPER_BIG_BRAIN:
            for _ in range(8):
                input_ids, output_ids, masks = next(it)  
                input_ids = input_ids.to(device)
                model(input_ids, masks.view(-1, UDBB_SAMPLE_LEN, UDBB_SAMPLE_LEN)) 
        else:           
            for _ in range(8):
                input_ids, output_ids = next(it)  
                input_ids = input_ids.to(device)
                model(input_ids, None) 

    with torch.no_grad():
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        if data_mode == DataMode.ULTRA_DUPER_BIG_BRAIN:
            try:
                while True:
                    torch.cuda.synchronize()
                    starter.record()
                    input_ids, output_ids, masks = next(it)
                    input_ids = input_ids.to(device)
                    model(input_ids, masks.view(-1, UDBB_SAMPLE_LEN, UDBB_SAMPLE_LEN)) 
                    ender.record()
                    torch.cuda.synchronize()
                    batch_processing_time.append(starter.elapsed_time(ender))
            except StopIteration:
                pass
        else:
            try:
                while True:
                    torch.cuda.synchronize()
                    starter.record()
                    input_ids, output_ids = next(it)
                    input_ids = input_ids.to(device)
                    model(input_ids, None) 
                    ender.record()
                    torch.cuda.synchronize()
                    batch_processing_time.append(starter.elapsed_time(ender))
            except StopIteration:
                pass

    logs[dataset_name]['min'] = np.min(batch_processing_time)
    logs[dataset_name]['max'] = np.max(batch_processing_time)
    logs[dataset_name]['mean'] = np.mean(batch_processing_time)
    logs[dataset_name]['median'] = np.median(batch_processing_time)
    logs[dataset_name]['sample'] = batch_processing_time

