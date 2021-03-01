"""

@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

from __future__ import absolute_import, division, print_function

import os
import numpy as np

import tensorflow as tf
import torch
from torch import nn, optim
from torch.nn.parallel.data_parallel import DataParallel

from models.pytorchwrapper import WeakDefense as PyTorchWD
from models.networks import get_model, num_class

tf.compat.v1.disable_eager_execution()

IMAGE_SHAPE = (3, 32, 32)

def load_model(file, model_configs,
               trans_configs=None, gpu_device=None):
    if torch.cuda.is_available():
        if gpu_device and gpu_device.startswith("cuda:"):
            gpu_device = gpu_device
        else:
            gpu_device = "cuda:0"
        torch.cuda.set_device(torch.device(gpu_device))
        device = torch.device(gpu_device)
    else:
        device = torch.device("cpu")

    if trans_configs is None:
        # load the undefended model by default
        trans_configs = {
            "type": "clean",
            "subtype": "",
            "id": 0,
            "description": "clean"
        }

    network_configs = model_configs.get("network_configs")
    model = get_model(
        model_type=network_configs.get("model_type"),
        num_class=num_class(model_configs.get("dataset")),
        data_parallel=torch.cuda.is_available(),
        device=device
    )

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=network_configs.get("lr"),
        momentum=network_configs.get("optimizer_momentum"),
        weight_decay=network_configs.get("optimizer_decay"),
        nesterov=network_configs.get("optimizer_nesterov")
    )
    
    if os.path.isfile(file):
        print(">>> Loading model from [{}]...".format(file))
        data = torch.load(file, map_location=lambda storage, loc: storage)

        if 'model' in data or 'state_dict' in data:
            key = 'model' if 'model' in data else 'state_dict'
            print('checkpoint epoch@%d' % data['epoch'])
            if not isinstance(model, DataParallel):
                model.load_state_dict({k.replace('module.', ''): v for k, v in data[key].items()})
            else:
                model.load_state_dict({k if 'module.' in k else 'module.'+k: v for k, v in data[key].items()})
            optimizer.load_state_dict(data['optimizer'])
        else:
            model.load_state_dict({k: v for k, v in data.items()})
        del data
    else:
        raise ValueError(file, 'is not found.')

    model.eval()
    if model_configs.get("wrap", True):
        print("Wrap model")
        model = PyTorchWD(
            model=model,
            loss=loss_func,
            optimizer=optimizer,
            input_shape=IMAGE_SHAPE,
            nb_classes=num_class(model_configs.get("dataset")),
            trans_configs=trans_configs,
            channel_index=1,
            clip_values=(0., 1.)
        )

    return model, loss_func, optimizer


def load_pool(trans_configs, pool_name, model_configs, active_list=False):
    pool = {}
    trans_list = {}

    if active_list:
        # load the specific pool
        wd_ids = trans_configs.get(pool_name)
    else:
        # load the full pool
        num_trans = trans_configs.get("num_transformations")
        wd_ids = [i for i in range(num_trans)]

    for i in wd_ids:
        key = "configs{}".format(i)
        trans = trans_configs.get(key).get("description")

        # models are named in the form of
        # <transformatin>-<dataset>-<network_configs.model_name>.<file_format>
        # e.g., filter_rank-cifar100-wres28x10.pth
        model_file = "{}-{}_{}.{}".format(
            trans,
            model_configs.get("dataset"),
            model_configs.get("network_configs").get("model_name"),
            model_configs.get("file_format")
        )
        model_file = os.path.join(model_configs.get("dir"), model_file)
        wd, _, _ = load_model(
            file=model_file,
            model_configs=model_configs,
            trans_configs=trans_configs.get(key)
        )
        pool[trans_configs.get(key).get("id")] = wd
        trans_list[trans_configs.get(key).get("id")] = trans

    print(">>> Loaded {} models.".format(len(pool.keys())))
    return pool, trans_list
