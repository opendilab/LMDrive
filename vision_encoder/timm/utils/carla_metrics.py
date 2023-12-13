""" Eval metrics and related

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch


def l1_accuracy(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    batch_size = target.size(0)
    length = target.size(1)
    return torch.sum(torch.abs(output - target)) / (batch_size * length)
