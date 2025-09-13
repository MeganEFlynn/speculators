
import argparse
import json
import os
import random
import warnings
from typing import Any

import numpy as np
import safetensors
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from model.configs import EConfig
from model.llama_eagle3_full_grad import Model
from safetensors import safe_open
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, get_linear_schedule_with_warmup

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data
class DataCollatorWithPadding:
    def paddingtensor(self, intensors, dim):
        b, n, s = intensors.shape
        padding_tensor = torch.zeros(b, dim - n, s)
        return torch.cat((intensors, padding_tensor), dim=1)

    def paddingtensor2d(self, intensors, num):
        b, n = intensors.shape

        padding_tensor = torch.zeros(b, num - n, dtype=intensors.dtype)

        return torch.cat((intensors, padding_tensor), dim=1)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        max_length = max(item["hidden_state_big"].shape[1] for item in features)

        batch_input_ids = torch.cat(
            [self.paddingtensor2d(item["input_ids"], max_length) for item in features]
        )
        batch_hidden_states = torch.cat(
            [
                self.paddingtensor(item["hidden_state_big"], max_length)
                for item in features
            ]
        )
        batch_target = torch.cat(
            [self.paddingtensor(item["target"], max_length) for item in features]
        )
        batch_loss_mask = torch.tensor(
            [
                item["loss_mask"] + [0] * (max_length - len(item["loss_mask"]))
                for item in features
            ]
        )
        batch_attention_mask = torch.tensor(
            [
                item["attention_mask"]
                + [0] * (max_length - len(item["attention_mask"]))
                for item in features
            ]
        )
        return {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }


def top_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res


def compute_loss(target_p, predict, loss_mask, kldiv=None):
    out_head = predict
    out_logp = nn.LogSoftmax(dim=2)(out_head)
    kldiv_loss = kldiv(out_logp, target_p)
    kldiv_loss = torch.sum(torch.sum(loss_mask * kldiv_loss, 2)) / (
        loss_mask.sum() + 1e-5
    )
    return out_head, kldiv_loss
