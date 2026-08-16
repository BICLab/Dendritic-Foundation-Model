import os
import sys

import math
import bisect
import random
import pandas as pd

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transforms import find_first, Repeat, toOneHot, ToTensor
from events_timeslices import chunk_evs_pol_dvs_gait, get_tmad_slice
# from datasets.event_drop import *

def random_shift_events(events, max_shift=20, resolution=(128, 128), p=0.5):
    H, W = resolution
    if np.random.random() < p:
        x_shift, y_shift = np.random.randint(-max_shift, max_shift+1, size=(2, ))
        events[:, 1] += x_shift
        events[:, 2] += y_shift
        valid_events = (events[:, 1] >= 0) & (events[:, 1] < W) & (events[:, 2] >= 0) & (events[:, 2] < H)
        events = events[valid_events]
    return events



class DVS128GaitDataset(Dataset):
    def __init__(
        self,
        path,
        dt=1000,
        T=10,
        train=True,
        is_train_Enhanced=False,
        clips=1,
        is_spike=False,
        ds=None,
        save_event_data_path=None,
        load_clip_select=None
    ):
        super(DVS128GaitDataset, self).__init__()
        if ds is None:
            ds = [1, 1]
        self.train = train
        self.dt = dt
        self.T = T
        self.is_train_Enhanced = is_train_Enhanced
        self.clips = clips
        self.is_spike = is_spike
        self.ds = ds
        self.save_event_data_path = save_event_data_path
        self.load_clip_select = load_clip_select

        if self.train:
            train_npy_path = os.path.join(path, 'train')
            self.train_data = np.load(os.path.join(train_npy_path, 'train_data.npy'), allow_pickle=True)
            self.train_target = np.load(os.path.join(train_npy_path, 'train_target.npy'), allow_pickle=True)
        else:
            test_npy_path = os.path.join(path, 'test')
            self.test_data = np.load(os.path.join(test_npy_path, 'test_data.npy'), allow_pickle=True)
            self.test_target = np.load(os.path.join(test_npy_path, 'test_target.npy'), allow_pickle=True)

            if load_clip_select:
                self.load_clip_select = pd.read_csv(load_clip_select).values[0][1:]
            else:
                self.load_clip_select = np.ones(len(self.test_data))*-1

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, idx):
        if self.train:
            data = self.train_data[idx]

            # data = random_shift_events(data)

            data = sample_train(
                data=data,
                dt=self.dt,
                T=self.T,
                is_train_Enhanced=self.is_train_Enhanced,
            )

            data = chunk_evs_pol_dvs_gait(
                data=data,
                dt=self.dt,
                T=self.T,
                ds=self.ds
            )
            # if self.is_spike:
            #     data = np.int64(data > 0)

            target_idx = self.train_target[idx]
            # label = np.zeros((20))
            # label[target_idx] = 1.0
            data = np.float32(data)

            return data, target_idx
        else:
            data = self.test_data[idx]

            data = sample_test(
                data=data,
                dt=self.dt,
                T=self.T,
                idx=idx,
                clips=self.clips,
                save_event_data_path=self.save_event_data_path,
                load_clip_select=self.load_clip_select[idx]
            )

            target_idx = self.test_target[idx]

            data_temp = []
            target_temp = []
            for i in range(self.clips):
                temp = chunk_evs_pol_dvs_gait(
                    data=data[i],
                    dt=self.dt,
                    T=self.T,
                    ds=self.ds
                )

                data_temp.append(temp)

            data = np.float32(np.array(data_temp))
            
            return data, target_idx


def sample_train(
    data,
    T=60,
    dt=1000,
    is_train_Enhanced=False
):
    tbegin = data[:, 0][0]
    tend = np.maximum(0, data[:, 0][-1] - T * dt)

    start_time = random.randint(tbegin, tend) if is_train_Enhanced else tbegin

    tmad = get_tmad_slice(
        data[:, 0],
        data[:, 1:4],
        start_time,
        T * dt
    )
    if len(tmad) == 0:
        return tmad
    tmad[:, 0] -= tmad[0, 0]
    return tmad


def sample_test(
    data,
    T=60,
    clips=10,
    dt=1000,
    save_event_data_path=None,
    idx=None,
    load_clip_select=None,
):
    tbegin = data[:, 0][0]
    tend = np.maximum(0, data[:, 0][-1])

    tmad = get_tmad_slice(
        data[:, 0],
        data[:, 1:4],
        tbegin,
        tend - tbegin
    )
    # 初试从零开始
    tmad[:, 0] -= tmad[0, 0]

    start_time = tmad[0, 0]
    end_time = tmad[-1, 0]

    start_point = []
    if clips * T * dt - (end_time - start_time) > 0:
        overlap = int(
            np.floor((clips * T * dt - (end_time - start_time)) / clips))
        for j in range(clips):
            start_point.append(j * (T * dt - overlap))
            if start_point[-1] + T * dt > end_time:
                diff = start_point[-1] + T * dt - end_time
                start_point[-1] = start_point[-1] - diff
    else:
        overlap = int(
            np.floor(((end_time - start_time) - clips * T * dt) / clips))
        for j in range(clips):
            start_point.append(j * (T * dt + overlap))
            if start_point[-1] + T * dt > end_time:
                diff = start_point[-1] + T * dt - end_time
                start_point[-1] = start_point[-1] - diff

    temp = []
    for start in start_point:
        idx_beg = find_first(tmad[:, 0], start)
        idx_end = find_first(tmad[:, 0][idx_beg:], start + T * dt) + idx_beg

        if save_event_data_path:
            if start_point.index(start) == load_clip_select:
                save_data = tmad[idx_beg:idx_end][:, :]
                save_data = torch.tensor(save_data.astype(int))
                save_data[:,1] = torch.floor(save_data[:,1]/4)
                save_data[:,2] = torch.floor(save_data[:,2]/4)
                save_data = save_data.detach().numpy()
                np.savetxt(save_event_data_path + 'idx_'+str(idx) + '_clip_' + str(load_clip_select) + '.txt',save_data[:, [0, 3, 2, 1]],fmt='%d')

        temp.append(tmad[idx_beg:idx_end])

    return temp


def create_gait_datasets(
    root=None,
    train=True,
    chunk_size_train=60,
    chunk_size_test=60,
    ds=4,
    dt=1000,
    transform_train=None,
    transform_test=None,
    target_transform_train=None,
    target_transform_test=None,
    n_events_attention=None,
    clip=10,
    is_train_Enhanced=False,
    is_spike=False,
    interval_scaling=False,
    T=16,
    save_event_data_path=None,
    load_clip_select=None,
):
    if isinstance(ds, int):
        ds = [ds, ds]

    if n_events_attention is None:
        def default_transform():
            return Compose([
                ToTensor()
            ])
    else:
        def default_transform():
            return Compose([
                ToTensor()
            ])

    if transform_train is None:
        transform_train = default_transform()
    if transform_test is None:
        transform_test = default_transform()

    if target_transform_train is None:
        target_transform_train = Compose(
            [Repeat(chunk_size_train), toOneHot(11)])
    if target_transform_test is None:
        target_transform_test = Compose(
            [Repeat(chunk_size_test), toOneHot(11)])

    dataset = DVS128GaitDataset(
        root,
        dt=dt,
        T=T,
        train=train,
        is_train_Enhanced=is_train_Enhanced,
        clips=clip,
        is_spike=is_spike,
        ds=ds,
        save_event_data_path=save_event_data_path,
        load_clip_select=load_clip_select
    )
    return dataset
