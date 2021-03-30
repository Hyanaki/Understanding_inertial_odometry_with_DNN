import os
import numpy as np
import torch
import matplotlib.pyplot as plt


def prepare_data(args, dataset, dataset_name, idx_start=None, idx_end=None, to_numpy=False):
    # get data
    t, ang_gt, p_gt, v_gt, u = dataset.get_data(dataset_name)

    # get start instant
    if idx_start is None:
        idx_start = 0
    # get end instant
    if idx_end is None:
        idx_end = t.shape[0]

    t = t[idx_start: idx_end]
    u = u[idx_start: idx_end]
    ang_gt = ang_gt[idx_start: idx_end]
    v_gt = v_gt[idx_start: idx_end]
    p_gt = p_gt[idx_start: idx_end] - p_gt[idx_start]

    if to_numpy:
        t = t.cpu().double().numpy()
        u = u.cpu().double().numpy()
        ang_gt = ang_gt.cpu().double().numpy()
        v_gt = v_gt.cpu().double().numpy()
        p_gt = p_gt.cpu().double().numpy()
    return t, ang_gt, p_gt, v_gt, u


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)

