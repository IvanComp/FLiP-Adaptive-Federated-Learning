import json
import pickle
import os
import random
import time
from collections import Counter
from collections import OrderedDict
from logging import INFO
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.utils as vutils
from flwr.common.logger import log
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, TensorDataset, ConcatDataset
from torch.utils.data import Dataset
from torchgan.losses import MinimaxGeneratorLoss, MinimaxDiscriminatorLoss
from torchgan.models import DCGANGenerator, DCGANDiscriminator
from torchgan.trainer import Trainer
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST, KMNIST, OxfordIIITPet
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose, ToPILImage
from torchtext.datasets import IMDB as TorchTextIMDB
from torchtext.datasets import YahooAnswers as TorchTextYahooAnswers
from torchtext.datasets import AG_NEWS as TorchTextAGNews
from torchtext.data.utils import get_tokenizer

from torchtext.vocab import build_vocab_from_iterator


class TensorLabelDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if not torch.is_tensor(y):
            y = torch.tensor(y)
        return x, y


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GLOBAL_ROUND_COUNTER = 1
HGAN_DONE = False
global CLIENT_SELECTOR, CLIENT_CLUSTER, MESSAGE_COMPRESSOR, MULTI_TASK_MODEL_TRAINER, HETEROGENEOUS_DATA_HANDLER
CLIENT_SELECTOR = False
CLIENT_CLUSTER = False
MESSAGE_COMPRESSOR = False
MULTI_TASK_MODEL_TRAINER = False
HETEROGENEOUS_DATA_HANDLER = False
global DATASET_TYPE, DATASET_NAME
DATASET_TYPE = ""
DATASET_NAME = ""
AVAILABLE_DATASETS = {
    "CIFAR10": {
        "class": CIFAR10,
        "normalize": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        "channels": 3,
        "num_classes": 10
    },
    "CIFAR100": {
        "class": CIFAR100,
        "normalize": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        "channels": 3,
        "num_classes": 100
    },
    "MNIST": {
        "class": MNIST,
        "normalize": ((0.5,), (0.5,)),
        "channels": 1,
        "num_classes": 10
    },
    "FashionMNIST": {
        "class": FashionMNIST,
        "normalize": ((0.5,), (0.5,)),
        "channels": 1,
        "num_classes": 10
    },
    "KMNIST": {
        "class": KMNIST,
        "normalize": ((0.5,), (0.5,)),
        "channels": 1,
        "num_classes": 10
    },
    "ImageNet100": {
        "class": None,
        "normalize": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "channels": 3,
        "num_classes": 10
    },
    "OXFORDIIITPET": {
        "class": OxfordIIITPet,
        "normalize": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "channels": 3,
        "num_classes": 37
    },
    "IMDB": {
        "class": TorchTextIMDB,
        "normalize": ((0.0,), (1.0,)),
        "channels": 1,
        "num_classes": 2
    },
    "YAHOOANSWERS": {
        "class": TorchTextYahooAnswers,
        "normalize": ((0.0,), (1.0,)),
        "channels": 1,
        "num_classes": 10
    },
    "AG_NEWS": {
        "class": TorchTextAGNews,
        "normalize": ((0.0,), (1.0,)),
        "channels": 1,
        "num_classes": 4
    },
    "SST2": {
        "class": None,
        "normalize": ((0.0,), (1.0,)),
        "channels": 1,
        "num_classes": 2
    },

    "DBPEDIA": {
        "class": None,
        "normalize": ((0.0,), (1.0,)),
        "channels": 1,
        "num_classes": 14
    },
    "SPEECHCOMMANDS": {
        "class": None,
        "normalize": None,
        "channels": 1,
        "num_classes": 30
    },
    "YESNO": {
        "class": None,
        "normalize": None,
        "channels": 1,
        "num_classes": 2
    },
    "CMUARCTIC": {
        "class": None,
        "normalize": None,
        "channels": 1,
        "num_classes": 4
    },
    "PROSTATE_CANCER": {
        "class": None,
        "normalize": None,
        "channels": 1,
        "num_classes": 2,
        "num_features": 8
    }
}
_orig_make_grid = vutils.make_grid


def make_grid_no_range(*args, **kwargs):
    kwargs.pop("range", None)
    return _orig_make_grid(*args, **kwargs)


vutils.make_grid = make_grid_no_range
current_dir = os.path.abspath(os.path.dirname(__file__))
config_dir = os.path.join(current_dir, 'configuration')
config_file = os.path.join(config_dir, 'config.json')


def get_valid_downscale_size(size: int) -> int:
    power = 32
    while power * 2 <= size and power * 2 <= 128:
        power *= 2
    return power


def normalize_dataset_name(name: str) -> str:
    name_clean = name.replace("-", "").upper()
    if name_clean == "CIFAR10":
        return "CIFAR10"
    elif name_clean == "CIFAR100":
        return "CIFAR100"
    elif name_clean == "IMAGENET100":
        return "ImageNet100"
    elif name_clean == "MNIST":
        return "MNIST"
    elif name_clean == "FASHIONMNIST":
        return "FashionMNIST"
    elif name_clean == "FMNIST":
        return "FMNIST"
    elif name_clean == "KMNIST":
        return "KMNIST"
    elif name_clean == "OXFORDIIITPET":
        return "OXFORDIIITPET"
    elif name_clean == "IMDB":
        return "IMDB"
    elif name_clean in ("YAHOOANSWERS", "YAHOO_ANSWERS", "YAHOO"):
        return "YAHOOANSWERS"
    elif name_clean in ("AG_NEWS", "AGNEWS"):
        return "AG_NEWS"
    elif name_clean in ("AG_NEWS", "AGNEWS"):
        return "AG_NEWS"
    elif name_clean == "SPEECHCOMMANDS":
        return "SPEECHCOMMANDS"
    elif name_clean == "YESNO":
        return "YESNO"
    elif name_clean == "CMUARCTIC":
        return "CMUARCTIC"
    elif name_clean in ("PROSTATECANCER", "PROSTATE", "PCA", "PROSTATEFL"):
        return "PROSTATE_CANCER"
    else:
        return name


if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        configJSON = json.load(f)
    for pattern_name, pattern_info in configJSON["patterns"].items():
        if pattern_info["enabled"]:
            if pattern_name == "client_selector":
                CLIENT_SELECTOR = True
            elif pattern_name == "client_cluster":
                CLIENT_CLUSTER = True
            elif pattern_name == "message_compressor":
                MESSAGE_COMPRESSOR = True
            elif pattern_name == "multi-task_model_trainer":
                MULTI_TASK_MODEL_TRAINER = True
            elif pattern_name == "heterogeneous_data_handler":
                HETEROGENEOUS_DATA_HANDLER = True
    ds = configJSON.get("dataset") or configJSON["client_details"][0].get("dataset", None)
    if ds is None:
        raise ValueError(
            "Il file di configurazione non specifica il dataset né tramite la chiave 'dataset' né in 'client_details'.")
    DATASET_NAME = normalize_dataset_name(ds)
    DATASET_TYPE = configJSON["client_details"][0].get("data_distribution_type", "")


class CNN_Dynamic(nn.Module):
    def __init__(
            self, num_classes: int, input_size: int, in_ch: int,
            conv1_out: int, conv2_out: int, fc1_out: int, fc2_out: int
    ) -> None:
        super(CNN_Dynamic, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, conv1_out, kernel_size=5)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        dummy = torch.zeros(1, in_ch, input_size, input_size)
        dummy = self.pool(F.relu(self.conv1(dummy)))
        dummy = self.pool(F.relu(self.conv2(dummy)))
        flat_size = dummy.view(1, -1).size(1)
        self.fc1 = nn.Linear(flat_size, fc1_out)
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc3 = nn.Linear(fc2_out, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class TextMLP(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int) -> None:
        super(TextMLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        h = emb.mean(dim=1)
        h = F.relu(self.fc1(h))
        return self.fc2(h)


class TextLSTM(nn.Module):
    """LSTM-based model for text classification."""

    def __init__(
            self,
            vocab_size: int,
            embed_dim: int,
            hidden_dim: int,
            num_classes: int,
            num_layers: int = 2,
            bidirectional: bool = True,
            dropout: float = 0.3
    ) -> None:
        super(TextLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(emb)
        if self.lstm.bidirectional:
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            hidden_combined = torch.cat((hidden_forward, hidden_backward), dim=1)
        else:
            hidden_combined = hidden[-1, :, :]
        return self.fc(hidden_combined)


class ProstateLogisticRegression(nn.Module):
    """Single-logit binary logistic regression for the prostate task."""

    def __init__(self, input_dim: int = 8, num_classes: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

        # Use a deterministic, non-zero initialization for this task. A zero
        # initialization makes the first full-batch logistic-regression step
        # immediately recover almost the final decision direction, which yields
        # an artificially flat accuracy curve. The fixed seed makes repeated FL
        # runs comparable while leaving all other task initializations unchanged.
        rng = np.random.default_rng()
        bound = 1.0 / np.sqrt(float(input_dim))
        init_w = rng.uniform(-bound, bound, size=(1, input_dim)).astype(np.float32)
        init_b = rng.uniform(-bound, bound, size=(1,)).astype(np.float32)
        with torch.no_grad():
            self.linear.weight.copy_(torch.from_numpy(init_w))
            self.linear.bias.copy_(torch.from_numpy(init_b))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.float()).squeeze(-1)


class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2).squeeze(1)


def get_weight_class_dynamic(model_name: str):
    weight_mapping = {
        "cnn": None,
        "alexnet": "AlexNet_Weights",
        "convnext_tiny": "ConvNeXt_Tiny_Weights",
        "convnext_small": "ConvNeXt_Small_Weights",
        "convnext_base": "ConvNeXt_Base_Weights",
        "convnext_large": "ConvNeXt_Large_Weights",
        "densenet121": "DenseNet121_Weights",
        "densenet161": "DenseNet161_Weights",
        "densenet169": "DenseNet169_Weights",
        "densenet201": "DenseNet201_Weights",
        "efficientnet_b0": "EfficientNet_B0_Weights",
        "efficientnet_b1": "EfficientNet_B1_Weights",
        "efficientnet_b2": "EfficientNet_B2_Weights",
        "efficientnet_b3": "EfficientNet_B3_Weights",
        "efficientnet_b4": "EfficientNet_B4_Weights",
        "efficientnet_b5": "EfficientNet_B5_Weights",
        "efficientnet_b6": "EfficientNet_B6_Weights",
        "efficientnet_b7": "EfficientNet_B7_Weights",
        "efficientnet_v2_s": "EfficientNet_V2_S_Weights",
        "efficientnet_v2_m": "EfficientNet_V2_M_Weights",
        "efficientnet_v2_l": "EfficientNet_V2_L_Weights",
        "googlenet": "GoogLeNet_Weights",
        "inception_v3": "Inception_V3_Weights",
        "mnasnet0_5": "MnasNet0_5_Weights",
        "mnasnet0_75": "MnasNet0_75_Weights",
        "mnasnet1_0": "MnasNet1_0_Weights",
        "mnasnet1_3": "MnasNet1_3_Weights",
        "mobilenet_v2": "MobileNet_V2_Weights",
        "mobilenet_v3_large": "MobileNet_V3_Large_Weights",
        "mobilenet_v3_small": "MobileNet_V3_Small_Weights",
        "regnet_x_400mf": "RegNet_X_400MF_Weights",
        "regnet_x_800mf": "RegNet_X_800MF_Weights",
        "regnet_x_1_6gf": "RegNet_X_1_6GF_Weights",
        "regnet_x_16gf": "RegNet_X_16GF_Weights",
        "regnet_x_32gf": "RegNet_X_32GF_Weights",
        "regnet_x_3_2gf": "RegNet_X_3_2GF_Weights",
        "regnet_x_8gf": "RegNet_X_8GF_Weights",
        "regnet_y_400mf": "RegNet_Y_400MF_Weights",
        "regnet_y_800mf": "RegNet_Y_800MF_Weights",
        "regnet_y_128gf": "RegNet_Y_128GF_Weights",
        "regnet_y_16gf": "RegNet_Y_16GF_Weights",
        "regnet_y_1_6gf": "RegNet_Y_1_6GF_Weights",
        "regnet_y_32gf": "RegNet_Y_32GF_Weights",
        "regnet_y_3_2gf": "RegNet_Y_3_2GF_Weights",
        "regnet_y_8gf": "RegNet_Y_8GF_Weights",
        "resnet18": "ResNet18_Weights",
        "resnet34": "ResNet34_Weights",
        "resnet50": "ResNet50_Weights",
        "resnet101": "ResNet101_Weights",
        "resnet152": "ResNet152_Weights",
        "resnext50_32x4d": "ResNeXt50_32X4D_Weights",
        "shufflenet_v2_x0_5": "ShuffleNet_V2_x0_5_Weights",
        "shufflenet_v2_x1_0": "ShuffleNet_V2_x1_0_Weights",
        "squeezenet1_0": "SqueezeNet1_0_Weights",
        "squeezenet1_1": "SqueezeNet1_1_Weights",
        "vgg11": "VGG11_Weights",
        "vgg11_bn": "VGG11_BN_Weights",
        "vgg13": "VGG13_Weights",
        "vgg13_bn": "VGG13_BN_Weights",
        "vgg16": "VGG16_Weights",
        "vgg16_bn": "VGG16_BN_Weights",
        "vgg19": "VGG19_Weights",
        "vgg19_bn": "VGG19_BN_Weights",
        "wide_resnet50_2": "Wide_ResNet50_2_Weights",
        "wide_resnet101_2": "Wide_ResNet101_2_Weights",
        "swin_t": "Swin_T_Weights",
        "swin_s": "Swin_S_Weights",
        "swin_b": "Swin_B_Weights",
        "vit_b_16": "ViT_B_16_Weights",
        "vit_b_32": "ViT_B_32_Weights",
        "vit_l_16": "ViT_L_16_Weights",
        "vit_l_32": "ViT_L_32_Weights"
    }
    model_name = model_name.lower()
    weight_class_name = weight_mapping.get(model_name, None)
    if weight_class_name is not None:
        return getattr(models, weight_class_name, None)
    return None


def get_dynamic_model(num_classes: int, model_name: str = None, pretrained: bool = True) -> nn.Module:
    if model_name is None:
        with open(config_file, 'r') as f:
            configJSON = json.load(f)
        model_name = configJSON["client_details"][0].get("model")
    name = model_name.strip().lower().replace("-", "_").replace(" ", "_")
    if name in ("cnn_16k", "cnn16k"):
        input_size = {
            "CIFAR10": 32, "CIFAR100": 32,
            "FashionMNIST": 28, "MNIST": 28, "KMNIST": 28, "FMNIST": 28,
            "ImageNet100": 224, "OXFORDIIITPET": 224
        }[DATASET_NAME]
        in_ch = AVAILABLE_DATASETS[DATASET_NAME]["channels"]
        return CNN_Dynamic(
            num_classes, input_size, in_ch,
            conv1_out=3, conv2_out=8,
            fc1_out=60, fc2_out=42
        )
    if name in ("cnn_64k", "cnn64k"):
        input_size = {
            "CIFAR10": 32, "CIFAR100": 32,
            "FashionMNIST": 28, "MNIST": 28, "KMNIST": 28, "FMNIST": 28,
            "ImageNet100": 224, "OXFORDIIITPET": 224
        }[DATASET_NAME]
        in_ch = AVAILABLE_DATASETS[DATASET_NAME]["channels"]
        return CNN_Dynamic(
            num_classes, input_size, in_ch,
            conv1_out=6, conv2_out=16,
            fc1_out=120, fc2_out=84
        )
    if name in ("cnn_256k", "cnn256k"):
        input_size = {
            "CIFAR10": 32, "CIFAR100": 32,
            "FashionMNIST": 28, "MNIST": 28, "KMNIST": 28, "FMNIST": 28,
            "ImageNet100": 224, "OXFORDIIITPET": 224
        }[DATASET_NAME]
        in_ch = AVAILABLE_DATASETS[DATASET_NAME]["channels"]
        return CNN_Dynamic(
            num_classes, input_size, in_ch,
            conv1_out=12, conv2_out=32,
            fc1_out=240, fc2_out=168
        )
    if name in ("textmlp", "text_mlp"):
        return TextMLP(
            vocab_size=500,
            embed_dim=32,
            num_classes=num_classes,
        )
    if name in ("textlstm", "text_lstm", "lstm"):
        return TextLSTM(
            vocab_size=250,
            embed_dim=16,
            hidden_dim=16,
            num_classes=num_classes,
            num_layers=1,
            bidirectional=True,

            dropout=0.2,
        )
    if name in ("m5", "m_5"):
        return M5(n_input=1, n_output=num_classes)
    if name in ("prostate_logreg", "prostate_logistic_regression", "tabular_logreg", "logistic_regression"):
        return ProstateLogisticRegression(
            input_dim=AVAILABLE_DATASETS["PROSTATE_CANCER"]["num_features"],
            num_classes=num_classes,
        )
    if not hasattr(models, name):
        raise ValueError(f"Modello '{model_name}' non in torchvision.models")
    constructor = getattr(models, name)
    weight_cls = get_weight_class_dynamic(name)
    if pretrained and weight_cls and hasattr(weight_cls, "DEFAULT"):
        model = constructor(weights=weight_cls.DEFAULT, progress=False)
    else:
        model = constructor(weights=None, progress=False)
    if hasattr(model, "fc"):
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
    elif hasattr(model, "head"):
        in_f = model.head.in_features
        model.head = nn.Linear(in_f, num_classes)
    elif hasattr(model, "classifier"):
        cls = model.classifier
        if isinstance(cls, nn.Sequential):
            for i in reversed(range(len(cls))):
                m = cls[i]
                if isinstance(m, nn.Linear):
                    in_f = m.in_features
                    cls[i] = nn.Linear(in_f, num_classes)
                    break
                if isinstance(m, nn.Conv2d):
                    out_ch = m.out_channels
                    cls[i] = nn.Conv2d(m.in_channels, num_classes,
                                       kernel_size=m.kernel_size,
                                       stride=m.stride,
                                       padding=m.padding)
                    break
            model.classifier = cls
        else:
            in_f = cls.in_features
            model.classifier = nn.Linear(in_f, num_classes)
    else:
        raise NotImplementedError(f"{name} not Supported!")
    return model


def Net():
    with open(config_file, 'r') as f:
        configJSON = json.load(f)
    ds = configJSON.get("dataset", None)
    if ds is None:
        ds = configJSON["client_details"][0].get("dataset", None)
    dataset_name = normalize_dataset_name(ds)
    model_name = configJSON["client_details"][0].get("model", None)
    num_classes = AVAILABLE_DATASETS[dataset_name]["num_classes"]
    return get_dynamic_model(num_classes, model_name)


def get_non_iid_indices(dataset,
                        remove_class_frac,
                        add_class_frac,
                        remove_pct_range,
                        add_pct_range):
    cls2idx = {}
    for i, (_, lbl) in enumerate(dataset):
        cls2idx.setdefault(lbl, []).append(i)
    classes = list(cls2idx.keys())
    n_cls = len(classes)
    n_remove = max(1, int(remove_class_frac * n_cls))
    remove_cls = random.sample(classes, n_remove)
    avail = [c for c in classes if c not in remove_cls]
    raw_add = max(1, int(add_class_frac * n_cls))
    n_add = min(raw_add, len(avail))
    add_cls = random.sample(avail, n_add)
    pct_remove = {c: random.uniform(*remove_pct_range) for c in remove_cls}
    pct_add = {c: random.uniform(*add_pct_range) for c in add_cls}
    selected = []
    for c, idxs in cls2idx.items():
        n = len(idxs)
        if c in pct_remove:
            keep = int(n * (1 - pct_remove[c]))
            selected += random.sample(idxs, keep)
        elif c in pct_add:
            add_n = int(n * pct_add[c])
            selected += idxs + random.choices(idxs, k=add_n)
        else:
            selected += idxs
    total = len(dataset)
    if len(selected) > total:
        selected = random.sample(selected, total)
    elif len(selected) < total:
        selected += random.choices(selected, k=total - len(selected))
    zero_cls = random.choice(classes)
    selected = [i for i in selected if dataset[i][1] != zero_cls]
    if len(selected) > total:
        selected = random.sample(selected, total)
    elif len(selected) < total:
        selected += random.choices(selected, k=total - len(selected))
    return selected
    return selected


def load_balanced_data(iterator, max_total, num_classes, label_extractor, item_processor):
    """
    Loads a balanced subset of data from a generator/iterator.
    Ensures each class gets approximately max_total // num_classes samples.
    """
    counts = defaultdict(int)
    target = max_total // num_classes
    data = []
    filled_classes = 0
    for raw_item in iterator:
        lbl = label_extractor(raw_item)
        if counts[lbl] < target:
            processed_item = item_processor(raw_item)
            data.append(processed_item)
            counts[lbl] += 1
            if counts[lbl] == target:
                filled_classes += 1
        if filled_classes == num_classes:
            break
    random.seed(42)
    random.shuffle(data)
    return data


PROSTATE_FEATURE_COLUMNS = [
    "age", "PSA", "PV", "PSA_SPLINE_2", "PSA_SPLINE_3",
    "PIRADS_3", "PIRADS_4", "PIRADS_5",
]


def _prostate_spline(x: float, i: int) -> float:
    """Restricted cubic spline used by the source prostate-cancer study."""
    t = [3.80, 6.60, 9.40, 18.47]
    return (
        max(x - t[i], 0) ** 3
        - max(x - t[2], 0) ** 3 * (t[3] - t[i]) / (t[3] - t[2])
        + max(x - t[3], 0) ** 3 * (t[2] - t[i]) / (t[3] - t[2])
    )


def _prepare_prostate_dataframe(df):
    """Convert one preprocessed hospital dataframe to the paper's model features."""
    required = {"sig_cancer", "age", "PSA", "PV", "PIRADS"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Prostate dataset is missing required columns: {sorted(missing)}")

    value = df.copy()
    if "5ARI" not in value.columns:
        value["5ARI"] = 0.0

    for col in ["sig_cancer", "age", "PSA", "PV", "PIRADS", "5ARI"]:
        value[col] = pd.to_numeric(value[col], errors="coerce")
    value = value.dropna(subset=["sig_cancer", "age", "PSA", "PV", "PIRADS"])

    # Same 5-alpha-reductase-inhibitor correction and spline transformation
    # used in the public replication package of Kazlouski et al. (2025).
    on_5ari = value["5ARI"] == 1
    value.loc[on_5ari, "PV"] = value.loc[on_5ari, "PV"] / 0.7
    value.loc[on_5ari, "PSA"] = value.loc[on_5ari, "PSA"] * 2.0
    value["PSA_SPLINE_2"] = value["PSA"].apply(lambda x: _prostate_spline(float(x), 0))
    value["PSA_SPLINE_3"] = value["PSA"].apply(lambda x: _prostate_spline(float(x), 1))

    pirads = value["PIRADS"].astype(int)
    value["PIRADS_3"] = (pirads == 3).astype(float)
    value["PIRADS_4"] = (pirads == 4).astype(float)
    value["PIRADS_5"] = (pirads == 5).astype(float)
    value["sig_cancer"] = value["sig_cancer"].astype(int)

    value = value[["sig_cancer"] + PROSTATE_FEATURE_COLUMNS]
    value = value.replace([np.inf, -np.inf], np.nan).dropna()
    return value.reset_index(drop=True)


def _stratified_frame_split(df, test_fraction: float, seed: int):
    """Match the source study's sklearn stratified 80/20 silo split."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_fraction,
        random_state=seed,
        stratify=df["sig_cancer"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _allocate_prostate_shards(silo_sizes, total_clients: int):
    """Allocate virtual clients across source hospitals using largest remainders."""
    names = list(silo_sizes.keys())
    if total_clients < 1:
        raise ValueError("The prostate task requires at least one client.")
    if total_clients < len(names):
        names = sorted(names, key=lambda n: silo_sizes[n], reverse=True)[:total_clients]
        return {name: 1 for name in names}

    allocation = {name: 1 for name in names}
    remaining = total_clients - len(names)
    if remaining == 0:
        return allocation

    total_samples = float(sum(silo_sizes[name] for name in names))
    exact = {name: remaining * silo_sizes[name] / total_samples for name in names}
    floors = {name: int(np.floor(exact[name])) for name in names}
    for name in names:
        allocation[name] += floors[name]
    left = remaining - sum(floors.values())
    order = sorted(names, key=lambda n: (exact[n] - floors[n], silo_sizes[n]), reverse=True)
    for name in order[:left]:
        allocation[name] += 1
    return allocation


def _split_prostate_silo_into_shards(df, n_shards: int, seed: int):
    """Split one hospital's training data into stratified, non-overlapping virtual clients."""
    if n_shards == 1:
        return [df.reset_index(drop=True)]
    shards = [[] for _ in range(n_shards)]
    rng = np.random.default_rng(seed)
    for label in sorted(df["sig_cancer"].unique()):
        idx = df.index[df["sig_cancer"] == label].to_numpy()
        idx = rng.permutation(idx)
        for shard_id, part in enumerate(np.array_split(idx, n_shards)):
            shards[shard_id].extend(part.tolist())
    return [df.loc[idx].sample(frac=1.0, random_state=seed + i).reset_index(drop=True)
            for i, idx in enumerate(shards)]


def _frame_to_tensor_dataset(df, feature_mean=None, feature_std=None):
    x = torch.tensor(df[PROSTATE_FEATURE_COLUMNS].to_numpy(dtype=np.float32), dtype=torch.float32)
    y = torch.tensor(df["sig_cancer"].to_numpy(dtype=np.int64), dtype=torch.long)
    dataset = TensorDataset(x, y)

    # Metadata is used only by the prostate-specific HDH implementation to
    # reconstruct the original clinical variables before applying SMOTENC.
    # Attaching it to the dataset leaves all other task and loader behavior
    # unchanged, including Subset-based "New Data" inflow.
    if feature_mean is not None and feature_std is not None:
        dataset.prostate_feature_mean = np.asarray(feature_mean, dtype=np.float64)
        dataset.prostate_feature_std = np.asarray(feature_std, dtype=np.float64)

    return dataset


def _standardize_prostate_dataframe(df, feature_mean, feature_std):
    """Apply training-only feature standardization for stable iterative LR."""
    value = df.copy()
    value.loc[:, PROSTATE_FEATURE_COLUMNS] = (
        value[PROSTATE_FEATURE_COLUMNS] - feature_mean
    ) / feature_std
    return value


PROSTATE_DATA_URL = (
    "https://raw.githubusercontent.com/phaseivai/"
    "Towards-Practical-FL/main/data/data.pkl"
)


def _ensure_prostate_dataset(data_path: Path) -> Path:
    """Download the source study's preprocessed data.pkl once if it is missing."""
    if data_path.exists():
        return data_path

    import fcntl
    import urllib.request

    data_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = data_path.parent / ".prostate_data_download.lock"

    # All Docker clients mount the same ./data directory. The lock prevents
    # simultaneous downloads when many clients start at the same time.
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if data_path.exists():
                return data_path

            tmp_path = data_path.with_suffix(data_path.suffix + ".tmp")
            if tmp_path.exists():
                tmp_path.unlink()

            log(INFO, f"Downloading prostate dataset to {data_path}...")
            try:
                urllib.request.urlretrieve(PROSTATE_DATA_URL, tmp_path)
                if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                    raise RuntimeError("Downloaded prostate dataset is empty.")

                # Validate the pickle before exposing it to the other clients.
                with open(tmp_path, "rb") as f:
                    downloaded = pickle.load(f)
                if not isinstance(downloaded, dict) or not downloaded:
                    raise ValueError(
                        "Downloaded prostate data.pkl does not contain the expected silo dictionary."
                    )

                os.replace(tmp_path, data_path)
                log(INFO, f"Prostate dataset downloaded successfully ({data_path.stat().st_size} bytes).")
            except Exception as exc:
                if tmp_path.exists():
                    tmp_path.unlink()
                raise RuntimeError(
                    f"Could not download the prostate dataset from {PROSTATE_DATA_URL}: {exc}"
                ) from exc
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return data_path


def _load_prostate_cross_silo_data(client_config):
    """
    Load the prostate-cancer task from the public replication package.

    The source data.pkl contains natural hospital/country silos. For a federation
    larger than the 11 natural silos used in the paper, each hospital's training
    partition is deterministically subdivided into non-overlapping virtual silos;
    samples from different source hospitals are never mixed within a client.
    """
    configured_path = client_config.get("prostate_data_path")
    if configured_path:
        data_path = Path(configured_path)
    else:
        data_path = Path(current_dir) / "data" / "prostate" / "data.pkl"

    data_path = _ensure_prostate_dataset(data_path)

    with open(data_path, "rb") as f:
        silos_raw = pickle.load(f)
    if not isinstance(silos_raw, dict) or not silos_raw:
        raise ValueError("The prostate data.pkl file must contain a non-empty dictionary of silo DataFrames.")

    # The source study excludes these two datasets from its multi-silo experiments.
    excluded = set(client_config.get("prostate_excluded_silos", ["Turkey", "Finland"]))
    silos = {}
    for name, df in silos_raw.items():
        if name in excluded:
            continue
        silos[name] = _prepare_prostate_dataframe(df)
    if not silos:
        raise ValueError("No prostate silos remain after applying prostate_excluded_silos.")

    # The original implementation seeds NumPy with 1234, draws one split
    # random_state, and reuses that same state for every hospital. Reproduce
    # that behavior by default (1234 -> 486191), while allowing an explicit
    # override for controlled sensitivity experiments.
    master_seed = int(client_config.get("prostate_master_seed", 1234))
    default_split_seed = int(np.random.RandomState(master_seed).randint(0, 10000000))
    split_seed = int(client_config.get("prostate_split_seed", default_split_seed))
    test_fraction = float(client_config.get("prostate_test_fraction", 0.20))
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("prostate_test_fraction must be between 0 and 1.")

    train_silos, test_silos = {}, {}
    for name in sorted(silos):
        train_silos[name], test_silos[name] = _stratified_frame_split(
            silos[name], test_fraction=test_fraction, seed=split_seed
        )

    with open(config_file, "r") as f:
        full_config = json.load(f)
    total_clients = int(client_config.get("prostate_num_clients", len(full_config.get("client_details", []))))
    if total_clients <= 0:
        total_clients = len(train_silos)

    allocation = _allocate_prostate_shards(
        {name: len(df) for name, df in train_silos.items()}, total_clients
    )

    # Logistic regression is still the same linear model, but the raw spline
    # features differ by several orders of magnitude. For a genuinely iterative
    # 20-round optimization, standardize predictors using statistics computed
    # only from the participating training partitions. The same deterministic
    # transformation is then applied to every participating training/test silo.
    # This is benchmark preprocessing; no validation samples contribute to it.
    participating_silos = sorted(allocation)
    stats_df = pd.concat([train_silos[name] for name in participating_silos], ignore_index=True)
    feature_mean = stats_df[PROSTATE_FEATURE_COLUMNS].mean()
    feature_std = stats_df[PROSTATE_FEATURE_COLUMNS].std(ddof=0).replace(0.0, 1.0)
    for name in participating_silos:
        train_silos[name] = _standardize_prostate_dataframe(
            train_silos[name], feature_mean, feature_std
        )
        test_silos[name] = _standardize_prostate_dataframe(
            test_silos[name], feature_mean, feature_std
        )

    virtual_clients = []
    for offset, name in enumerate(sorted(allocation)):
        shards = _split_prostate_silo_into_shards(
            train_silos[name], allocation[name], seed=split_seed + 1000 + offset
        )
        virtual_clients.extend((name, shard_id, shard) for shard_id, shard in enumerate(shards))

    # Client IDs in setup.py determine simulated resource classes (high-spec
    # clients are created first). Randomize the data-to-ID assignment with a
    # fixed seed so hospital identity/size is not systematically coupled to
    # the assigned CPU class in the 40-client experiment.
    assignment_seed = int(client_config.get("prostate_client_assignment_seed", 1234))
    assignment_rng = np.random.default_rng(assignment_seed)
    assignment_rng.shuffle(virtual_clients)

    client_id = int(os.environ.get("CLIENT_ID", client_config.get("client_id", 1)))
    client_index = client_id - 1
    if client_index < 0 or client_index >= len(virtual_clients):
        raise ValueError(
            f"CLIENT_ID={client_id} is outside the configured prostate federation "
            f"of {len(virtual_clients)} clients."
        )

    source_silo, shard_id, client_df = virtual_clients[client_index]
    if client_df.empty:
        raise ValueError(
            f"Prostate client {client_id} ({source_silo}, shard {shard_id}) has no training samples."
        )

    # If fewer than the 11 natural silos are requested for a smoke test, only
    # evaluate on the held-out portions of the silos that actually participate
    # in training. For 11+ clients, all 11 source silos are represented.
    test_df = pd.concat([test_silos[name] for name in participating_silos], ignore_index=True)
    log(INFO, f"Prostate client {client_id}: source_silo={source_silo}, shard={shard_id}, "
              f"train_samples={len(client_df)}, global_test_samples={len(test_df)}, "
              f"split_seed={split_seed}")
    return (
        _frame_to_tensor_dataset(client_df, feature_mean, feature_std),
        _frame_to_tensor_dataset(test_df, feature_mean, feature_std),
    )


def load_data(client_config, GLOBAL_ROUND_COUNTER, dataset_name_override=None):
    global DATASET_NAME, DATASET_TYPE, DATASET_PERSISTENCE
    DATASET_TYPE = client_config.get("data_distribution_type", "").lower()
    DATASET_PERSISTENCE = client_config.get("data_persistence_type", "")
    dataset_name = dataset_name_override or client_config.get("dataset", "")
    DATASET_NAME = normalize_dataset_name(dataset_name)
    if DATASET_NAME == "IMDB":
        MAX_SAMPLES = 25000
    elif DATASET_NAME == "AG_NEWS":
        MAX_SAMPLES = 120000
        MAX_SAMPLES = 120000
    else:
        MAX_SAMPLES = 120000
    if DATASET_NAME not in AVAILABLE_DATASETS:
        raise ValueError(f"[ERROR] Dataset '{DATASET_NAME}' non trovato in AVAILABLE_DATASETS.")
    config = AVAILABLE_DATASETS[DATASET_NAME]
    normalize_params = config["normalize"]
    if DATASET_NAME == "PROSTATE_CANCER":
        trainset, testset = _load_prostate_cross_silo_data(client_config)
        batch_size = int(client_config.get("batch_size", 32))
    elif DATASET_NAME == "IMDB":
        tokenizer = get_tokenizer("basic_english")

        def _yield_tokens(data_iter):
            for label, text in data_iter:
                yield tokenizer(text)

        def _load_imdb_iterator(split: str):
            """Load IMDB dataset, trying different methods for compatibility."""
            try:
                from torchtext.datasets import IMDB
                import inspect
                sig = inspect.signature(IMDB)
                if 'root' in sig.parameters or 'split' in sig.parameters:
                    data_iter = IMDB(root="./data", split=split)
                    for item in data_iter:
                        yield item
            except (TypeError, ImportError):
                pass
            import tarfile
            import urllib.request
            import fcntl
            data_dir = Path("./data/aclImdb")
            lock_file = Path("./data/.imdb_download.lock")
            lock_file.parent.mkdir(parents=True, exist_ok=True)
            with open(lock_file, 'w') as lf:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                try:
                    if not data_dir.exists():
                        log(INFO, "Downloading IMDB dataset manually...")
                        url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
                        tar_path = Path("./data/aclImdb_v1.tar.gz")
                        tar_path.parent.mkdir(parents=True, exist_ok=True)
                        urllib.request.urlretrieve(url, tar_path)
                        with tarfile.open(tar_path, "r:gz") as tar:
                            tar.extractall("./data")
                        if tar_path.exists():
                            tar_path.unlink()
                        log(INFO, "IMDB dataset downloaded and extracted.")
                finally:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
            for sentiment in ["pos", "neg"]:
                folder = data_dir / split / sentiment
                label = 1 if sentiment == "pos" else 0
                if folder.exists():
                    for txt_file in sorted(folder.glob("*.txt")):
                        text = txt_file.read_text(encoding="utf-8")
                        yield (label, text)

        train_data_raw = _load_imdb_iterator("train")
        use_new_vocab_api = True
        try:
            vocab = build_vocab_from_iterator(_yield_tokens(train_data_raw), specials=["<unk>"])
        except TypeError:
            use_new_vocab_api = False
            from collections import Counter
            token_counter = Counter()
            for tokens in _yield_tokens(train_data_raw):
                token_counter.update(tokens)
            from torchtext.vocab import Vocab
            vocab = Vocab(token_counter, specials=["<unk>"])
        if hasattr(vocab, 'set_default_index'):
            vocab.set_default_index(vocab["<unk>"])
            unk_idx = vocab["<unk>"]
        else:
            unk_idx = vocab.stoi.get("<unk>", 0)
        max_len = 128
        model_name = client_config.get("model", "").strip().lower().replace("-", "_").replace(" ", "_")
        if model_name in ("textmlp", "text_mlp"):
            max_vocab = 500
        elif model_name in ("textlstm", "text_lstm", "lstm"):
            max_vocab = 250
        else:
            max_vocab = 20000

        def _get_token_idx(tok):
            """Get token index, handling old/new torchtext API."""
            if hasattr(vocab, '__getitem__') and use_new_vocab_api:
                try:
                    return vocab[tok]
                except KeyError:
                    return unk_idx
            else:
                return vocab.stoi.get(tok, unk_idx)

        def _encode_text(text: str) -> torch.Tensor:
            tokens = tokenizer(text)
            ids = []
            for tok in tokens[:max_len]:
                idx = _get_token_idx(tok)
                if idx >= max_vocab:
                    idx = unk_idx
                ids.append(idx)
            if len(ids) < max_len:
                ids += [0] * (max_len - len(ids))
            return torch.tensor(ids, dtype=torch.long)

        def _label_to_int(lbl):
            if isinstance(lbl, int):
                return lbl
            s = str(lbl).lower()
            if s in ("pos", "positive", "1", "2"):
                return 1
            else:
                return 0

        test_data_raw = _load_imdb_iterator("test")
        trainset = load_balanced_data(
            _load_imdb_iterator("train"),
            MAX_SAMPLES,
            2,
            lambda x: _label_to_int(x[0]),
            lambda x: (_encode_text(x[1]), _label_to_int(x[0]))
        )
        test_iter_raw = _load_imdb_iterator("test")
        testset = []
        for raw_label, text in test_iter_raw:
            x = _encode_text(text)
            y = _label_to_int(raw_label)
            testset.append((x, y))
        batch_size = int(client_config.get("batch_size", 64))
    elif DATASET_NAME == "YAHOOANSWERS":
        tokenizer = get_tokenizer("basic_english")

        def _yield_tokens(data_iter):
            for label, text in data_iter:
                yield tokenizer(text)

        def _load_yahoo_iterator(split: str):
            """Load Yahoo Answers dataset, trying different methods for compatibility."""
            import csv
            import fcntl
            try:
                from torchtext.datasets import YahooAnswers
                import inspect
                sig = inspect.signature(YahooAnswers)
                if 'root' in sig.parameters or 'split' in sig.parameters:
                    data_iter = YahooAnswers(root="./data", split=split)
                    for item in data_iter:
                        yield item
                    return
            except (TypeError, ImportError):
                pass
            data_dir = Path("./data/yahoo_answers_csv")
            lock_file = Path("./data/.yahoo_download.lock")
            lock_file.parent.mkdir(parents=True, exist_ok=True)
            with open(lock_file, 'w') as lf:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                try:
                    if not data_dir.exists():
                        log(INFO, "Downloading Yahoo Answers dataset manually...")
                        import tarfile
                        import urllib.request
                        url = "https://s3.amazonaws.com/fast-ai-nlp/yahoo_answers_csv.tgz"
                        tar_path = Path("./data/yahoo_answers_csv.tgz")
                        tar_path.parent.mkdir(parents=True, exist_ok=True)
                        urllib.request.urlretrieve(url, tar_path)
                        with tarfile.open(tar_path, "r:gz") as tar:
                            tar.extractall("./data")
                        if tar_path.exists():
                            tar_path.unlink()
                        log(INFO, "Yahoo Answers dataset downloaded and extracted.")
                finally:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
            csv_file = data_dir / f"{split}.csv"
            if csv_file.exists():
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 4:
                            label = int(row[0])
                            text = " ".join(row[1:])
                            yield (label, text)

        train_data_raw = _load_yahoo_iterator("train")
        use_new_vocab_api = True
        try:
            vocab = build_vocab_from_iterator(_yield_tokens(train_data_raw), specials=["<unk>"])
        except TypeError:
            use_new_vocab_api = False
            from collections import Counter
            token_counter = Counter()
            for tokens in _yield_tokens(train_data_raw):
                token_counter.update(tokens)
            from torchtext.vocab import Vocab
            vocab = Vocab(token_counter, specials=["<unk>"])
        if hasattr(vocab, 'set_default_index'):
            vocab.set_default_index(vocab["<unk>"])
            unk_idx = vocab["<unk>"]
        else:
            unk_idx = vocab.stoi.get("<unk>", 0)
        max_len = 256
        model_name = client_config.get("model", "").strip().lower().replace("-", "_").replace(" ", "_")
        if model_name in ("textmlp", "text_mlp"):
            max_vocab = 500
        elif model_name in ("textlstm", "text_lstm", "lstm"):
            max_vocab = 250
        else:
            max_vocab = 20000

        def _get_token_idx(tok):
            """Get token index, handling old/new torchtext API."""
            if hasattr(vocab, '__getitem__') and use_new_vocab_api:
                try:
                    return vocab[tok]
                except KeyError:
                    return unk_idx
            else:
                return vocab.stoi.get(tok, unk_idx)

        def _encode_text(text: str) -> torch.Tensor:
            tokens = tokenizer(text)
            ids = []
            for tok in tokens[:max_len]:
                idx = _get_token_idx(tok)
                if idx >= max_vocab:
                    idx = unk_idx
                ids.append(idx)
            if len(ids) < max_len:
                ids += [0] * (max_len - len(ids))
            return torch.tensor(ids, dtype=torch.long)

        def _label_to_int(lbl):
            if isinstance(lbl, int):
                return lbl - 1
            return int(lbl) - 1

        test_data_raw = _load_yahoo_iterator("test")
        trainset = load_balanced_data(
            _load_yahoo_iterator("train"),
            MAX_SAMPLES,
            10,
            lambda x: _label_to_int(x[0]),
            lambda x: (_encode_text(x[1]), _label_to_int(x[0]))
        )
        test_iter_raw = _load_yahoo_iterator("test")
        testset = []
        test_limit = 5000
        for i, (raw_label, text) in enumerate(test_iter_raw):
            if i >= test_limit: break
            x = _encode_text(text)
            y = _label_to_int(raw_label)
            testset.append((x, y))
        batch_size = int(client_config.get("batch_size", 64))
    elif DATASET_NAME == "AG_NEWS":
        tokenizer = get_tokenizer("basic_english")

        def _yield_tokens(data_iter):
            for label, text in data_iter:
                yield tokenizer(text)

        def _load_agnews_iterator(split: str):
            """Load AG_NEWS dataset."""
            try:
                from torchtext.datasets import AG_NEWS
                import inspect
                sig = inspect.signature(AG_NEWS)
                if 'root' in sig.parameters or 'split' in sig.parameters:
                    data_iter = AG_NEWS(root="./data", split=split)
                if 'root' in sig.parameters or 'split' in sig.parameters:
                    data_iter = AG_NEWS(root="./data", split=split)
                    for item in data_iter:
                        yield item
                    return
            except (TypeError, ImportError):
                pass
            import csv
            import fcntl
            import tarfile
            import urllib.request
            data_dir = Path("./data/ag_news_csv")
            lock_file = Path("./data/.agnews_download.lock")
            lock_file.parent.mkdir(parents=True, exist_ok=True)
            with open(lock_file, 'w') as lf:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                try:
                    if not data_dir.exists():
                        log(INFO, "Downloading AG_NEWS dataset manually (Fallback strategy like IMDB)...")
                        url = "https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz"
                        tar_path = Path("./data/ag_news_csv.tgz")
                        tar_path.parent.mkdir(parents=True, exist_ok=True)
                        urllib.request.urlretrieve(url, tar_path)
                        with tarfile.open(tar_path, "r:gz") as tar:
                            tar.extractall("./data")
                        if tar_path.exists():
                            tar_path.unlink()
                        log(INFO, "AG_NEWS dataset downloaded and extracted.")
                finally:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
            csv_file = data_dir / f"{split}.csv"
            if csv_file.exists():
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 3:
                            label = int(row[0])
                            text = " ".join(row[1:])
                            yield (label, text)
                    return
            raise RuntimeError("Could not load AG_NEWS via torchtext or manual fallback")

        try:
            iter_ = _load_agnews_iterator("train")
            train_data_raw = list(iter_)
        except Exception as e:
            log(INFO, f"FATAL ERROR loading AG_NEWS: {e}")
            raise e
        use_new_vocab_api = True
        try:
            vocab = build_vocab_from_iterator(_yield_tokens(train_data_raw), specials=["<unk>"])
        except TypeError:
            use_new_vocab_api = False
            from collections import Counter
            token_counter = Counter()
            for tokens in _yield_tokens(train_data_raw):
                token_counter.update(tokens)
            from torchtext.vocab import Vocab
            vocab = Vocab(token_counter, specials=["<unk>"])
        if hasattr(vocab, 'set_default_index'):
            vocab.set_default_index(vocab["<unk>"])
            unk_idx = vocab["<unk>"]
        else:
            unk_idx = vocab.stoi.get("<unk>", 0)
        max_len = 128
        model_name = client_config.get("model", "").strip().lower().replace("-", "_").replace(" ", "_")
        if model_name in ("textmlp", "text_mlp"):
            max_vocab = 500
        elif model_name in ("textlstm", "text_lstm", "lstm"):
            max_vocab = 250
        else:
            max_vocab = 20000

        def _get_token_idx(tok):
            if hasattr(vocab, '__getitem__') and use_new_vocab_api:
                try:
                    return vocab[tok]
                except KeyError:
                    return unk_idx
            else:
                return vocab.stoi.get(tok, unk_idx)

        def _encode_text(text: str) -> torch.Tensor:
            tokens = tokenizer(text)
            ids = []
            for tok in tokens[:max_len]:
                idx = _get_token_idx(tok)
                if idx >= max_vocab:
                    idx = unk_idx
                ids.append(idx)
            if len(ids) < max_len:
                ids += [0] * (max_len - len(ids))
            return torch.tensor(ids, dtype=torch.long)

        def _label_to_int(lbl):
            if isinstance(lbl, int):
                return lbl - 1
            return int(lbl) - 1

        test_data_raw = _load_agnews_iterator("test")
        trainset = load_balanced_data(
            _load_agnews_iterator("train"),
            MAX_SAMPLES,
            4,
            lambda x: int(x[0]) - 1,
            lambda x: (_encode_text(x[1]), int(x[0]) - 1)
        )
        try:
            test_iter_raw = _load_agnews_iterator("test")
        except:
            from torchtext.datasets import AG_NEWS
            test_iter_raw = AG_NEWS(root="./data", split="test")
        testset = []
        for raw_label, text in test_iter_raw:
            x = _encode_text(text)
            y = int(raw_label) - 1
            testset.append((x, y))
        batch_size = int(client_config.get("batch_size", 64))
    elif DATASET_NAME == "SST2":
        tokenizer = get_tokenizer("basic_english")

        def _yield_tokens(data_iter):
            for label, text in data_iter:
                yield tokenizer(text)

        def _load_sst2_iterator(split: str):
            """Load SST2 dataset with fallback strategy."""
            try:
                from torchtext.datasets import SST2
                import inspect
                sig = inspect.signature(SST2)
                if 'root' in sig.parameters or 'split' in sig.parameters:
                    data_iter = SST2(root="./data", split=split)
                    for item in data_iter:
                        if isinstance(item, tuple):
                            if isinstance(item[0], int):
                                yield item
                            else:
                                yield (item[1], item[0])
                    return
                datasets = SST2(root="./data")
                if isinstance(datasets, tuple):
                    if split == "train":
                        target = datasets[0]
                    elif split == "test":
                        target = datasets[2]
                    else:
                        target = datasets[1]
                    for item in target:
                        if isinstance(item, tuple):
                            if isinstance(item[0], int):
                                yield item
                            else:
                                yield (item[1], item[0])
                    return
            except Exception:
                pass
            import csv
            import fcntl
            import zipfile
            import urllib.request
            data_dir = Path("./data/SST-2")
            lock_file = Path("./data/.sst2_download.lock")
            lock_file.parent.mkdir(parents=True, exist_ok=True)
            with open(lock_file, 'w') as lf:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                try:
                    if not data_dir.exists():
                        log(INFO, "Downloading SST2 dataset manually...")
                        url = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
                        zip_path = Path("./data/SST-2.zip")
                        urllib.request.urlretrieve(url, zip_path)
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall("./data")
                        if zip_path.exists():
                            zip_path.unlink()
                        log(INFO, "SST2 dataset downloaded and extracted.")
                finally:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
            filename = "train.tsv" if split == "train" else "dev.tsv"
            tsv_path = data_dir / filename
            if tsv_path.exists():
                with open(tsv_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f, delimiter='\t')
                    next(reader)
                    for row in reader:
                        if len(row) >= 2:
                            text = row[0]
                            label = int(row[1])
                            yield (label, text)
                    return
            raise RuntimeError("Could not load SST2 via torchtext or manual fallback")

        MAX_SAMPLES = 70000
        try:
            iter_ = _load_sst2_iterator("train")
            train_data_raw = list(iter_)
        except Exception as e:
            raise e
        use_new_vocab_api = True
        try:
            vocab = build_vocab_from_iterator(_yield_tokens(train_data_raw), specials=["<unk>"])
        except TypeError:
            use_new_vocab_api = False
            from collections import Counter
            token_counter = Counter()
            for tokens in _yield_tokens(train_data_raw):
                token_counter.update(tokens)
            from torchtext.vocab import Vocab
            vocab = Vocab(token_counter, specials=["<unk>"])
        if hasattr(vocab, 'set_default_index'):
            vocab.set_default_index(vocab["<unk>"])
            unk_idx = vocab["<unk>"]
        else:
            unk_idx = vocab.stoi.get("<unk>", 0)
        max_len = 64
        model_name = client_config.get("model", "").strip().lower().replace("-", "_").replace(" ", "_")
        if model_name in ("textmlp", "text_mlp"):
            max_vocab = 500
        elif model_name in ("textlstm", "text_lstm", "lstm"):
            max_vocab = 250
        else:
            max_vocab = 20000

        def _get_token_idx(tok):
            if hasattr(vocab, '__getitem__') and use_new_vocab_api:
                try:
                    return vocab[tok]
                except KeyError:
                    return unk_idx
            else:
                return vocab.stoi.get(tok, unk_idx)

        def _encode_text(text: str) -> torch.Tensor:
            tokens = tokenizer(text)
            ids = []
            for tok in tokens[:max_len]:
                idx = _get_token_idx(tok)
                if idx >= max_vocab: idx = unk_idx
                ids.append(idx)
            if len(ids) < max_len:
                ids += [0] * (max_len - len(ids))
            return torch.tensor(ids, dtype=torch.long)

        trainset = load_balanced_data(
            _load_sst2_iterator("train"),
            MAX_SAMPLES,
            2,
            lambda x: int(x[0]),
            lambda x: (_encode_text(x[1]), int(x[0]))
        )
        test_iter_raw = _load_sst2_iterator("test")
        testset = []
        for raw_label, text in test_iter_raw:
            x = _encode_text(text)
            y = int(raw_label)
            testset.append((x, y))
        batch_size = int(client_config.get("batch_size", 64))
    elif DATASET_NAME == "DBPEDIA":
        tokenizer = get_tokenizer("basic_english")

        def _yield_tokens(data_iter):
            for label, text in data_iter:
                yield tokenizer(text)

        def _load_dbpedia_iterator(split: str):
            """Load DBpedia dataset with fallback strategy."""
            try:
                from torchtext.datasets import DBpedia
                import inspect
                sig = inspect.signature(DBpedia)
                if 'root' in sig.parameters or 'split' in sig.parameters:
                    data_iter = DBpedia(root="./data", split=split)
                    for item in data_iter:
                        if isinstance(item, tuple):
                            if isinstance(item[0], int):
                                yield item
                            else:
                                yield (item[1], item[0])
                    return
                datasets = DBpedia(root="./data")
                if isinstance(datasets, tuple):
                    target_iter = datasets[0] if split == "train" else datasets[1]
                    for item in target_iter:
                        yield item
                    return
            except Exception:
                pass
            import csv
            import fcntl
            import tarfile
            import urllib.request
            data_dir = Path("./data/dbpedia_csv")
            lock_file = Path("./data/.dbpedia_download.lock")
            lock_file.parent.mkdir(parents=True, exist_ok=True)
            with open(lock_file, 'w') as lf:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                try:
                    if not data_dir.exists():
                        log(INFO, "Downloading DBpedia dataset manually...")
                        url = "https://s3.amazonaws.com/fast-ai-nlp/dbpedia_csv.tgz"
                        tar_path = Path("./data/dbpedia_csv.tgz")
                        tar_path.parent.mkdir(parents=True, exist_ok=True)
                        urllib.request.urlretrieve(url, tar_path)
                        with tarfile.open(tar_path, "r:gz") as tar:
                            tar.extractall("./data")
                        if tar_path.exists():
                            tar_path.unlink()
                        log(INFO, "DBpedia dataset downloaded and extracted.")
                finally:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
            csv_file = data_dir / f"{split}.csv"
            if csv_file.exists():
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 3:
                            label = int(row[0])
                            text = " ".join(row[1:])
                            yield (label, text)
                    return
            raise RuntimeError("Could not load DBpedia via torchtext or manual fallback")

        MAX_SAMPLES = 140000
        try:
            iter_ = _load_dbpedia_iterator("train")
            train_data_raw = list(iter_)
        except Exception as e:
            raise e
        use_new_vocab_api = True
        try:
            vocab = build_vocab_from_iterator(_yield_tokens(train_data_raw), specials=["<unk>"])
        except TypeError:
            use_new_vocab_api = False
            from collections import Counter
            token_counter = Counter()
            for tokens in _yield_tokens(train_data_raw):
                token_counter.update(tokens)
            from torchtext.vocab import Vocab
            vocab = Vocab(token_counter, specials=["<unk>"])
        if hasattr(vocab, 'set_default_index'):
            vocab.set_default_index(vocab["<unk>"])
            unk_idx = vocab["<unk>"]
        else:
            unk_idx = vocab.stoi.get("<unk>", 0)
        max_len = 128
        model_name = client_config.get("model", "").strip().lower().replace("-", "_").replace(" ", "_")
        if model_name in ("textmlp", "text_mlp"):
            max_vocab = 500
        elif model_name in ("textlstm", "text_lstm", "lstm"):
            max_vocab = 250
        else:
            max_vocab = 20000

        def _get_token_idx(tok):
            if hasattr(vocab, '__getitem__') and use_new_vocab_api:
                try:
                    return vocab[tok]
                except KeyError:
                    return unk_idx
            else:
                return vocab.stoi.get(tok, unk_idx)

        def _encode_text(text: str) -> torch.Tensor:
            tokens = tokenizer(text)
            ids = []
            for tok in tokens[:max_len]:
                idx = _get_token_idx(tok)
                if idx >= max_vocab: idx = unk_idx
                ids.append(idx)
            if len(ids) < max_len:
                ids += [0] * (max_len - len(ids))
            return torch.tensor(ids, dtype=torch.long)

        trainset = load_balanced_data(
            _load_dbpedia_iterator("train"),
            MAX_SAMPLES,
            14,
            lambda x: int(x[0]) - 1,
            lambda x: (_encode_text(x[1]), int(x[0]) - 1)
        )
        test_iter_raw = _load_dbpedia_iterator("test")
        testset = []
        test_limit = 5000
        for i, (raw_label, text) in enumerate(test_iter_raw):
            if i >= test_limit: break
            x = _encode_text(text)
            y = int(raw_label) - 1
            testset.append((x, y))
        batch_size = int(client_config.get("batch_size", 64))
    elif DATASET_NAME == "YESNO":
        import torchaudio
        from torchaudio.datasets import YESNO

        data_root = "./data"
        if not os.path.exists(data_root):
            os.makedirs(data_root)

        try:
            full_ds = YESNO(root=data_root, download=True)
        except Exception as e:
            log(INFO, f"Error downloading YESNO: {e}")
            raise e

        max_len = 0
        for i in range(len(full_ds)):
            wf, _, _ = full_ds[i]
            if wf.shape[1] > max_len: max_len = wf.shape[1]

        target_len = max_len

        processed_data = []
        for i in range(len(full_ds)):
            wf, sr, labels = full_ds[i]
            if wf.shape[1] < target_len:
                wf = F.pad(wf, (0, target_len - wf.shape[1]))
            else:
                wf = wf[:, :target_len]

            y = labels[0]
            processed_data.append((wf, y))

        random.seed(42)
        random.shuffle(processed_data)

        split_idx = int(0.8 * len(processed_data))
        trainset = processed_data[:split_idx]
        testset = processed_data[split_idx:]

        batch_size = int(client_config.get("batch_size", 4))

        batch_size = int(client_config.get("batch_size", 4))

    elif DATASET_NAME == "CMUARCTIC":
        import torchaudio
        from torchaudio.datasets import CMUARCTIC

        data_root = "./data/cmu_arctic"
        if not os.path.exists(data_root):
            os.makedirs(data_root)

        speakers = ["slt", "bdl", "clb", "rms"]
        speaker_to_idx = {spk: i for i, spk in enumerate(speakers)}

        full_data = []
        for spk in speakers:
            try:
                ds = CMUARCTIC(root=data_root, url=spk, download=True)
                for i in range(len(ds)):
                    # torchaudio CMUARCTIC return signature varies by version.
                    # Usually: (waveform, sample_rate, transcript, utterance_id) -> 4 items
                    # Safer to just take the first item as waveform
                    item = ds[i]
                    waveform = item[0]
                    full_data.append((waveform, speaker_to_idx[spk]))
            except Exception as e:
                log(INFO, f"Error loading CMUARCTIC speaker {spk}: {e}")

        random.seed(42)
        random.shuffle(full_data)

        new_sample_rate = 2000
        transform = torchaudio.transforms.Resample(orig_freq=16000, new_freq=new_sample_rate)
        processed_data = []
        target_len = 2000

        for wf, lbl in full_data:
            wf = transform(wf)
            if wf.shape[1] < target_len:
                wf = F.pad(wf, (0, target_len - wf.shape[1]))
            else:
                wf = wf[:, :target_len]
            processed_data.append((wf, lbl))

        split_idx = int(0.8 * len(processed_data))
        trainset = processed_data[:split_idx]
        testset = processed_data[split_idx:]

        batch_size = int(client_config.get("batch_size", 4))

    elif DATASET_NAME == "SPEECHCOMMANDS":
        # But wait, we need to know the *exact* set of combined labels present.
        # Let's do a dynamic scan on training set to be safe and robust.

        train_iter = SubsetFSC("train")
        valid_iter = SubsetFSC("valid")
        test_iter = SubsetFSC("test")

        # Scan for labels
        # Dataset item: (waveform, sample_rate, file_name, speaker_id, transcription, action, object, location)
        # index 5, 6, 7 are action, object, location

        unique_labels = set()
        # We can scan train_iter. But it might be slow if it opens files.
        # FSC implementation in torchaudio usually reads a CSV. So it should be fast to just iterate the walker?
        # Actually torchaudio datasets use a walker list. We can cheat and access it if we want speed,
        # but let's just iterate a few or assume standard 31.
        # Let's use the standard list above? It might be safer to dynamically build from train Set.

        # Optimized scan:
        # subset._walker usually contains the metadata.
        # For FSC: _walker is a list of indices or filenames?
        # Let's just iterate.
        pass

        # Re-defining intents list based on common FSC usage
        # unique combinations of action, object, location

        known_intents = sorted([
            "change language-none-none", "activate-music-none", "deactivate-lights-bedroom",
            "increase-heat-kitchen", "decrease-heat-kitchen", "increase-heat-washroom",
            "decrease-heat-washroom", "increase-heat-bedroom", "decrease-heat-bedroom",  # Wait, bedroom heat?
            # Let's just collect them dynamically to avoid errors.
        ])

        # Dynamic collection
        train_samples = []
        for i in range(len(train_iter)):
            item = train_iter[i]
            # (waveform, sample_rate, transcript, speaker_id, action, object, location) -> verify signature
            # Torchaudio 2.2.1 signature: 
            # (waveform, sample_rate, fileid, speaker_id, transcription, action, object, location)

            action, obj, loc = item[5], item[6], item[7]
            label_str = f"{action}-{obj}-{loc}"
            unique_labels.add(label_str)
            train_samples.append((item[0], label_str))

        labels = sorted(list(unique_labels))
        label_to_index = {l: i for i, l in enumerate(labels)}

        new_sample_rate = 8000
        transform = torchaudio.transforms.Resample(orig_freq=16000, new_freq=new_sample_rate)

        MAX_SAMPLES = 20000

        def _process_waveform(waveform, target_sr=new_sample_rate):
            if waveform.shape[1] < target_sr:  # 1 sec
                pass
            # Resample? FSC is 16k usually.
            # waveform shape (C, T)

            # Since we don't know original SR easily without checking item (it's in item[1]),
            # let's assume 16k if standard.
            # Actually item has it.

            # Simplified: assume we get waveform resampled if needed.
            # Let's do it in the loop or collation.
            return waveform

        # Process train set
        trainset = []
        for wf, lbl_str in train_samples:  # train_samples collected above
            # Resample/Pad
            if wf.shape[1] < 16000:  # Original length check?
                pass

                # Resample
            wf = transform(wf)

            target_len = new_sample_rate
            if wf.shape[1] < target_len:
                wf = F.pad(wf, (0, target_len - wf.shape[1]))
            else:
                wf = wf[:, :target_len]

            if lbl_str in label_to_index:
                trainset.append((wf, label_to_index[lbl_str]))

        # Process test set
        testset = []
        test_limit = 2000
        for i in range(len(test_iter)):
            if i >= test_limit: break
            item = test_iter[i]
            wf = item[0]
            # sr = item[1]
            action, obj, loc = item[5], item[6], item[7]
            lbl_str = f"{action}-{obj}-{loc}"

            if lbl_str not in label_to_index: continue  # Should not happen if classes align

            wf = transform(wf)
            target_len = new_sample_rate
            if wf.shape[1] < target_len:
                wf = F.pad(wf, (0, target_len - wf.shape[1]))
            else:
                wf = wf[:, :target_len]

            testset.append((wf, label_to_index[lbl_str]))

        batch_size = int(client_config.get("batch_size", 64))

    elif DATASET_NAME == "SPEECHCOMMANDS":
        import torchaudio
        from torchaudio.datasets import SPEECHCOMMANDS

        class SubsetSC(SPEECHCOMMANDS):
            def __init__(self, subset: str = None):
                super().__init__("./data", download=True)

                def load_list(filename):
                    filepath = os.path.join(self._path, filename)
                    with open(filepath) as fileobj:
                        return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

                if subset == "validation":
                    self._walker = load_list("validation_list.txt")
                elif subset == "testing":
                    self._walker = load_list("testing_list.txt")
                elif subset == "training":
                    excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
                    excludes = set(excludes)
                    self._walker = [w for w in self._walker if w not in excludes]

        labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go',
                  'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven',
                  'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
        try:
            temp_train_iter = SubsetSC("training")
            labels = sorted(list(set(datapoint[2] for datapoint in temp_train_iter)))
        except:
            pass

        label_to_index = {label: index for index, label in enumerate(labels)}

        new_sample_rate = 8000
        transform = torchaudio.transforms.Resample(orig_freq=16000, new_freq=new_sample_rate)

        MAX_SAMPLES = 20000

        def _process_waveform(waveform):
            # waveform: (Channel, Time)
            if waveform.shape[1] < 16000:
                # It might be short, it's fine, resample handles it
                pass

            # Resample
            if 16000 != new_sample_rate:
                waveform = transform(waveform)

            # Pad/Truncate to 1s (8000 samples)
            target_len = new_sample_rate
            if waveform.shape[1] < target_len:
                waveform = F.pad(waveform, (0, target_len - waveform.shape[1]))
            else:
                waveform = waveform[:, :target_len]
            return waveform

        def _yield_speech_data(split):
            ds = SubsetSC(split)
            for i in range(len(ds)):
                yield ds[i]

        trainset = load_balanced_data(
            _yield_speech_data("training"),
            MAX_SAMPLES,
            len(labels),
            lambda x: label_to_index[x[2]],  # x: (waveform, sample_rate, label, ...)
            lambda x: (_process_waveform(x[0]), label_to_index[x[2]])
        )

        testset = []
        test_limit = 2000
        ds_test = SubsetSC("testing")
        for i in range(len(ds_test)):
            if i >= test_limit: break
            waveform, sample_rate, label, _, _ = ds_test[i]
            x = _process_waveform(waveform)
            y = label_to_index[label]
            testset.append((x, y))

        batch_size = int(client_config.get("batch_size", 64))
    else:
        base_size = {
            "CIFAR10": 32, "CIFAR100": 32, "MNIST": 28,
            "FashionMNIST": 28, "KMNIST": 28,
            "ImageNet100": 256, "OXFORDIIITPET": 256
        }[DATASET_NAME]
        model_name = client_config.get("model", "resnet18").lower()
        target_size = 256 if model_name in ["alexnet", "vgg11", "vgg13", "vgg16", "vgg19"] else base_size
        transforms_list = []
        if DATASET_NAME == "ImageNet100":
            transforms_list = [Resize(224), CenterCrop(224)]
            batch_size = 64
        else:
            if target_size != base_size:
                transforms_list.append(Resize((target_size, target_size)))
            batch_size = 64
        transforms_list += [ToTensor(), Normalize(*normalize_params)]
        trf = Compose(transforms_list)
        if DATASET_NAME == "ImageNet100":
            DATA_ROOT = Path(__file__).resolve().parent / "data"
            train_path = DATA_ROOT / "imagenet100-preprocessed" / "train"
            test_path = DATA_ROOT / "imagenet100-preprocessed" / "test"
            if not os.path.isdir(train_path) or not os.path.isdir(test_path):
                raise FileNotFoundError(
                    f"Dataset ImageNet100 non trovato in {train_path} e {test_path}"
                )
            trainset = ImageFolder(train_path, transform=trf)
            testset = ImageFolder(test_path, transform=trf)
        else:
            cls = config["class"]
            if DATASET_NAME == "OXFORDIIITPET":
                trainset = cls("./data", split="trainval", download=True, transform=trf)
                testset = cls("./data", split="test", download=True, transform=trf)
            else:
                trainset = cls("./data", train=True, download=True, transform=trf)
                testset = cls("./data", train=False, download=True, transform=trf)
    if DATASET_TYPE == "non-iid" and DATASET_NAME != "PROSTATE_CANCER":
        classes = list({lbl for _, lbl in trainset})
        n_cls = len(classes)
        remove_frac = random.uniform(1 / n_cls, (n_cls - 1) / n_cls)
        add_frac = random.uniform(0, (n_cls - 1) / n_cls)
        low_r, high_r = sorted((random.uniform(0.5, 1.0), random.uniform(0.5, 1.0)))
        low_a, high_a = sorted((random.uniform(0.5, 1.0), random.uniform(0.5, 1.0)))
        idxs = get_non_iid_indices(
            trainset,
            remove_frac,
            add_frac,
            (low_r, high_r),
            (low_a, high_a),
        )
        base = Subset(trainset, idxs)
        trainset = base
    config_path = os.path.join(os.path.dirname(__file__), 'configuration', 'config.json')
    with open(config_path, 'r') as f:
        total_rounds = json.load(f).get("rounds")
    if DATASET_PERSISTENCE == "Same Data":
        # log(INFO, f"Persistence: Same Data. Keeping full dataset ({len(trainset)} samples).")
        pass
    else:
        # log(INFO, f"Persistence: {DATASET_PERSISTENCE}. Round {GLOBAL_ROUND_COUNTER}/{total_rounds}. Initial Size: {len(trainset)}")
        class_to_indices = defaultdict(list)
        for idx in range(len(trainset)):
            _, label = trainset[idx]
            class_to_indices[int(label)].append(idx)
        selected_indices = []
        NON_IID_ROUNDS = True
        NON_IID_ALPHA = 0.30
        NON_IID_SEED = 1234
        cid_raw = int(os.environ.get("CLIENT_ID", "1"))
        cid0 = max(0, cid_raw - 1)
        n_cls = len(class_to_indices)
        R = int(total_rounds)
        round_idx = max(1, min(GLOBAL_ROUND_COUNTER, R))
        shape_now = None
        if NON_IID_ROUNDS and DATASET_PERSISTENCE in {"New Data", "Remove Data"}:
            rng = np.random.default_rng(NON_IID_SEED + cid0)
            inc = rng.dirichlet([NON_IID_ALPHA] * R, size=n_cls)
            if DATASET_PERSISTENCE == "New Data":
                shape_now = np.cumsum(inc, axis=1)[:, round_idx - 1]
                target_frac_total = round_idx / R
            else:
                m = R - round_idx + 1
                shape_now = inc[:, :m].sum(axis=1)
                target_frac_total = m / R
        else:
            if DATASET_PERSISTENCE == "New Data":
                target_frac_total = round_idx / R
            elif DATASET_PERSISTENCE == "Remove Data":
                target_frac_total = (R - round_idx + 1) / R
            else:
                target_frac_total = 1.0
        labels_sorted = sorted(class_to_indices)
        pools = {}
        caps = []
        for lab in labels_sorted:
            idxs_all = np.array(class_to_indices[lab])
            r = np.random.default_rng(NON_IID_SEED + int(lab) + 1000 * cid0)
            idxs_all = r.permutation(idxs_all)
            pool = idxs_all
            pools[lab] = pool
            caps.append(len(pool))
        caps = np.array(caps, dtype=np.int64)
        pool_total = int(caps.sum())

        T_target = int(np.clip(
            np.floor(pool_total * float(target_frac_total)),
            0,
            pool_total
        ))

        # Iterative prostate LR can train on a single-class local subset, which is a
        # valid consequence of non-IID data arrival. It cannot, however, train on an
        # empty subset. Ensure at least one sample is available whenever the client's
        # full local shard is non-empty.
        if (
            DATASET_NAME == "PROSTATE_CANCER"
            and DATASET_PERSISTENCE in {"New Data", "Remove Data"}
            and pool_total > 0
        ):
            T_target = max(1, T_target)

        if shape_now is None:
            raw = caps.astype(np.float64)
        else:
            raw = np.clip(shape_now, 0.0, 1.0) * caps

        def sum_at_scale(s: float) -> int:
            return int(np.floor(np.minimum(s * raw, caps)).sum())

        if raw.sum() == 0:
            scaled = np.zeros_like(raw, dtype=np.float64)
        else:
            lo, hi = 0.0, 1.0
            while sum_at_scale(hi) < T_target:
                hi *= 2.0
                if hi > 1e12:
                    break
            for _ in range(48):
                mid = 0.5 * (lo + hi)
                if sum_at_scale(mid) >= T_target:
                    hi = mid
                else:
                    lo = mid
            scaled = np.minimum(hi * raw, caps)
        base = np.floor(scaled).astype(np.int64)
        rem = T_target - int(base.sum())
        if rem > 0:
            frac = (scaled - base) if raw.sum() > 0 else np.ones_like(base, dtype=float)
            order = np.argsort(-frac)
            i, L = 0, len(base)
            while rem > 0 and L > 0:
                idx = order[i % L]
                if base[idx] < caps[idx]:
                    base[idx] += 1
                    rem -= 1
                i += 1
        elif rem < 0:
            frac = (scaled - base) if raw.sum() > 0 else np.zeros_like(base, dtype=float)
            order = np.argsort(frac)
            i, L = 0, len(base)
            while rem < 0 and L > 0:
                idx = order[i % L]
                if base[idx] > 0:
                    base[idx] -= 1
                    rem += 1
                i += 1
        for k, lab in enumerate(labels_sorted):
            n_take = int(base[k])
            if n_take > 0:
                selected_indices.extend(pools[lab][:n_take].tolist())
        trainset = Subset(trainset, selected_indices)
    trainloader = DataLoader(TensorLabelDataset(trainset), batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader


def truncate_dataset(dataset, max_per_class: int):
    counts = defaultdict(int)
    kept_indices = []
    for idx, (_, lbl) in enumerate(dataset):
        lbl = int(lbl)
        if counts[lbl] < max_per_class:
            kept_indices.append(idx)
            counts[lbl] += 1
    return Subset(dataset, kept_indices)


def balance_dataset_with_gan(
        trainset,
        num_classes,
        target_per_class=None,
        latent_dim=100,
        epochs=1,
        batch_size=32,
        device=DEVICE,
):
    counts = Counter(lbl.item() for _, lbl in trainset)
    total = len(trainset)
    if target_per_class is None:
        target_per_class = total // num_classes
    under_cls = [c for c, cnt in counts.items() if 0 < cnt < target_per_class]
    if not under_cls:
        return trainset
    idxs = [i for i, (_, lbl) in enumerate(trainset) if lbl in under_cls]
    sample_data = trainset[0][0]
    is_text_data = len(sample_data.shape) == 1
    if is_text_data:
        seq_len = sample_data.shape[0]
        vocab_size = max(int(x.max().item()) for x, _ in trainset) + 1
        embed_dim = 32
        hidden_dim = 64
        log(INFO, f"[HDH TextGAN] Applying TextGAN to rebalance classes: {under_cls}")
        log(INFO, f"[HDH TextGAN] Detected text data: seq_len={seq_len}, vocab_size={vocab_size}")

        class TextGenerator(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc_latent = nn.Linear(latent_dim, hidden_dim)
                self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
                self.fc_out = nn.Linear(hidden_dim, vocab_size)

            def forward(self, z):
                h = F.relu(self.fc_latent(z))
                h = h.unsqueeze(1).repeat(1, seq_len, 1)
                out, _ = self.lstm(h)
                logits = self.fc_out(out)
                # Return one-hot (approx) for gradient flow. Gumbel-Softmax with hard=True ensures discrete output in forward,
                # but valid gradients in backward.
                tokens = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
                return tokens  # Shape: (batch, seq_len, vocab_size)

        class TextDiscriminator(nn.Module):
            def __init__(self):
                super().__init__()
                # Use Linear instead of Embedding to handle soft/one-hot inputs from Generator
                # Mathematically, x @ W is equivalent to Embedding lookup if x is one-hot.
                self.fc_emb = nn.Linear(vocab_size, embed_dim, bias=False)
                self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
                self.fc = nn.Linear(hidden_dim, 1)

            def forward(self, x):
                # x shape: (batch, seq_len, vocab_size)
                emb = self.fc_emb(x)
                _, (h, _) = self.lstm(emb)
                out = self.fc(h.squeeze(0))
                return torch.sigmoid(out)

        generator = TextGenerator().to(device)
        discriminator = TextDiscriminator().to(device)
        g_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        criterion = nn.BCELoss()
        subset_data = [trainset[i] for i in idxs]
        loader = DataLoader(subset_data, batch_size=batch_size, shuffle=True, drop_last=True)
        log(INFO, "[HDH TextGAN] Starting TextGAN training...")
        for epoch in range(epochs):
            for batch_idx, (real_x, _) in enumerate(loader):
                if batch_idx >= 20:  # Cap training to 20 batches to prevent CPU starvation/hangs
                    break

                if batch_idx % 5 == 0:
                    log(INFO, f"[HDH TextGAN] Epoch {epoch + 1}, Batch {batch_idx}...")
                real_x = real_x.to(device)
                bs = real_x.size(0)
                d_optimizer.zero_grad()
                real_labels = torch.ones(bs, 1, device=device)
                fake_labels = torch.zeros(bs, 1, device=device)

                # Convert real indices to one-hot for the new Discriminator
                real_x_onehot = F.one_hot(real_x, num_classes=vocab_size).float()
                d_real = discriminator(real_x_onehot)
                d_loss_real = criterion(d_real, real_labels)
                z = torch.randn(bs, latent_dim, device=device)
                fake_x = generator(z).detach()
                d_fake = discriminator(fake_x)
                d_loss_fake = criterion(d_fake, fake_labels)
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_optimizer.step()
                g_optimizer.zero_grad()
                z = torch.randn(bs, latent_dim, device=device)
                fake_x = generator(z)
                d_fake = discriminator(fake_x)
                g_loss = criterion(d_fake, real_labels)
                g_loss.backward()
                g_optimizer.step()
            log(INFO, f"[HDH TextGAN] Epoch {epoch + 1}/{epochs} completed.")
        synth_texts, synth_lbls = [], []
        generator.eval()
        for c in under_cls:
            cnt = counts[c]
            to_gen = target_per_class - cnt
            if to_gen <= 0:
                continue

            gen_batch_size = 32
            num_batches = (to_gen + gen_batch_size - 1) // gen_batch_size
            for i in range(num_batches):
                current_batch_size = min(gen_batch_size, to_gen - i * gen_batch_size)
                z = torch.randn(current_batch_size, latent_dim, device=device)
                with torch.no_grad():
                    gen_soft = generator(z).cpu()
                    gen_indices = gen_soft.argmax(dim=-1)  # Convert back to indices for the dataset
                synth_texts.append(gen_indices)

            synth_lbls += [c] * to_gen
        if synth_texts:
            all_texts = torch.cat(synth_texts, dim=0)
            all_lbls = torch.tensor(synth_lbls, dtype=torch.long)
            synth_ds = TensorDataset(all_texts, all_lbls)
            result = ConcatDataset([trainset, synth_ds])
            log(INFO, f"[HDH TextGAN] TextGAN Training Completed.")
            log(INFO, f"[HDH TextGAN] Rebalanced dataset size: {len(result)} (added {len(synth_lbls)} samples)")
            return result
        return trainset
    else:
        C, H, W = sample_data.shape
        target_size = get_valid_downscale_size(min(H, W))
        if H != target_size or W != target_size:
            resize_for_gan = Compose([
                ToPILImage(),
                Resize((target_size, target_size)),
                ToTensor(),
            ])
            train_for_gan = [(resize_for_gan(img), lbl) for img, lbl in trainset]
        else:
            train_for_gan = list(trainset)
        log(INFO, f"[HDH GAN] Applying GAN to rebalance classes: {under_cls}")
        subset = Subset(train_for_gan, idxs)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        import torchvision
        _orig_make_grid = torchvision.utils.make_grid

        def _make_grid_wrapper(*args, **kwargs):
            if 'range' in kwargs:
                kwargs['value_range'] = kwargs.pop('range')
            return _orig_make_grid(*args, **kwargs)

        torchvision.utils.make_grid = _make_grid_wrapper
        models_cfg = {
            'generator': {'name': DCGANGenerator,
                          'args': {'encoding_dims': latent_dim, 'out_size': target_size, 'out_channels': C},
                          'optimizer': {'name': torch.optim.Adam, 'args': {'lr': 2e-4, 'betas': (0.5, 0.999)}}},
            'discriminator': {'name': DCGANDiscriminator, 'args': {'in_size': target_size, 'in_channels': C},
                              'optimizer': {'name': torch.optim.Adam, 'args': {'lr': 2e-4, 'betas': (0.5, 0.999)}}},
        }
        losses = [MinimaxGeneratorLoss(), MinimaxDiscriminatorLoss()]
        log(INFO, "[HDH GAN] Starting GAN training...")
        trainer = Trainer(models=models_cfg, losses_list=losses, device=device, sample_size=batch_size, epochs=epochs)

        # Manually train with limits to prevent hangs
        # Trainer.train() loops over the whole loader. We need to override or just use a short loader.
        # Simpler: Subset the loader or just trust Image GAN is faster (usually is).
        # But for safety, let's limit the loader size itself before passing to trainer?
        # TorchGAN trainer doesn't support easy step limiting.
        # Let's reduce the dataset size passed to it if it's too large.

        if len(subset) > batch_size * 20:
            subset = Subset(train_for_gan, idxs[:batch_size * 20])
            loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

        trainer.train(loader)
        synth_imgs, synth_lbls = [], []
        for c in under_cls:
            cnt = counts[c]
            to_gen = target_per_class - cnt
            if to_gen <= 0:
                continue

            gen_batch_size = 32
            num_batches = (to_gen + gen_batch_size - 1) // gen_batch_size
            for i in range(num_batches):
                current_batch_size = min(gen_batch_size, to_gen - i * gen_batch_size)
                z = torch.randn(current_batch_size, latent_dim, device=device)
                with torch.no_grad():
                    gen = trainer.generator(z).cpu()
                synth_imgs.append(gen)

            synth_lbls += [c] * to_gen
        if synth_imgs:
            all_imgs_gan = torch.cat(synth_imgs, dim=0)
            downsample = Compose([ToPILImage(), Resize((H, W)), ToTensor()])
            resized = torch.stack([downsample(img) for img in all_imgs_gan])
            all_lbls = torch.tensor(synth_lbls, dtype=torch.long)
            synth_ds = TensorDataset(resized, all_lbls)
            result = ConcatDataset([trainset, synth_ds])
            log(INFO, f"[HDH GAN] GAN Training Completed.")
            log(INFO, f"[HDH GAN] Rebalanced dataset size: {len(result)} (added {len(synth_lbls)} samples)")
            return result
        return trainset


def _find_dataset_metadata(dataset, attribute):
    """Find metadata through TensorLabelDataset/Subset/ConcatDataset wrappers."""
    pending = [dataset]
    visited = set()

    while pending:
        current = pending.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))

        if hasattr(current, attribute):
            return getattr(current, attribute)
        if hasattr(current, "dataset"):
            pending.append(current.dataset)
        if hasattr(current, "datasets"):
            pending.extend(list(current.datasets))

    return None


def _prostate_loader_to_numpy(trainloader):
    """Extract the current prostate subset in deterministic dataset order."""
    features, labels = [], []
    dataset = trainloader.dataset

    for idx in range(len(dataset)):
        x, y = dataset[idx]
        x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        y_value = int(y.item()) if isinstance(y, torch.Tensor) else int(y)
        features.append(np.asarray(x_np, dtype=np.float64))
        labels.append(y_value)

    if not features:
        return np.empty((0, len(PROSTATE_FEATURE_COLUMNS)), dtype=np.float64), np.empty(0, dtype=np.int64)

    return np.stack(features, axis=0), np.asarray(labels, dtype=np.int64)


def _standardized_prostate_to_compact_raw(x_standardized, feature_mean, feature_std):
    """Recover age/PSA/PV/PI-RADS from the standardized model features."""
    x_raw = x_standardized * feature_std + feature_mean

    pirads_one_hot = x_raw[:, 5:8]
    pirads = np.zeros(len(x_raw), dtype=np.int64)
    if len(x_raw) > 0:
        selected = np.argmax(pirads_one_hot, axis=1)
        has_category = np.max(pirads_one_hot, axis=1) >= 0.5
        pirads[has_category] = selected[has_category] + 3

    return np.column_stack((x_raw[:, 0], x_raw[:, 1], x_raw[:, 2], pirads))


def _compact_raw_to_standardized_prostate(x_compact, y, feature_mean, feature_std):
    """Recompute dependent prostate features and restore the model representation."""
    value = pd.DataFrame(
        {
            "age": np.asarray(x_compact[:, 0], dtype=np.float64),
            "PSA": np.asarray(x_compact[:, 1], dtype=np.float64),
            "PV": np.asarray(x_compact[:, 2], dtype=np.float64),
            "PIRADS": np.rint(x_compact[:, 3]).astype(np.int64),
            "sig_cancer": np.asarray(y, dtype=np.int64),
        }
    )

    # SMOTENC interpolates only age/PSA/PV and treats PI-RADS as categorical.
    # Recompute all dependent variables so generated records remain internally
    # consistent with the feature engineering used by the source study.
    value["PSA_SPLINE_2"] = value["PSA"].apply(lambda x: _prostate_spline(float(x), 0))
    value["PSA_SPLINE_3"] = value["PSA"].apply(lambda x: _prostate_spline(float(x), 1))
    value["PIRADS_3"] = (value["PIRADS"] == 3).astype(float)
    value["PIRADS_4"] = (value["PIRADS"] == 4).astype(float)
    value["PIRADS_5"] = (value["PIRADS"] == 5).astype(float)

    value.loc[:, PROSTATE_FEATURE_COLUMNS] = (
        value[PROSTATE_FEATURE_COLUMNS] - feature_mean
    ) / feature_std
    return value[["sig_cancer"] + PROSTATE_FEATURE_COLUMNS]


def rebalance_trainloader_with_smotenc(trainloader):
    """
    Prostate-only HDH implementation based on SMOTENC.

    Continuous age/PSA/PV values are interpolated while PI-RADS is treated as
    categorical. The spline and one-hot features are recomputed afterwards.
    The final local dataset is balanced without changing its original size
    (apart from dropping one sample when the size is odd), matching the size
    control applied by the existing GAN-based HDH implementations.
    """
    start_time = time.time()

    # Lazy import is intentional: all non-prostate tasks remain independent of
    # imbalanced-learn and continue through the existing GAN path unchanged.
    try:
        from imblearn.over_sampling import SMOTENC
    except ImportError as exc:
        raise ImportError(
            "The prostate HDH requires imbalanced-learn. Install it with "
            "'pip install imbalanced-learn'."
        ) from exc

    x_standardized, y = _prostate_loader_to_numpy(trainloader)
    if len(y) == 0:
        return trainloader, 0.0

    class_counts = Counter(y.tolist())
    if len(class_counts) < 2:
        elapsed = time.time() - start_time
        log(INFO, "[HDH SMOTENC] Skipped: the current prostate subset contains only one class.")
        return trainloader, elapsed

    feature_mean = _find_dataset_metadata(trainloader.dataset, "prostate_feature_mean")
    feature_std = _find_dataset_metadata(trainloader.dataset, "prostate_feature_std")
    if feature_mean is None or feature_std is None:
        raise RuntimeError(
            "Missing prostate standardization metadata required by the SMOTENC HDH."
        )

    feature_mean = np.asarray(feature_mean, dtype=np.float64)
    feature_std = np.asarray(feature_std, dtype=np.float64)
    if feature_mean.shape != (len(PROSTATE_FEATURE_COLUMNS),) or feature_std.shape != feature_mean.shape:
        raise RuntimeError("Invalid prostate standardization metadata shape.")

    majority_label, majority_count = max(class_counts.items(), key=lambda item: item[1])
    minority_label, minority_count = min(class_counts.items(), key=lambda item: item[1])
    target_per_class = len(y) // 2

    # A meaningful SMOTE neighborhood needs at least two minority examples.
    # A single-class or singleton-minority early batch is left untouched rather
    # than fabricating unsupported clinical records.
    if minority_count < 2 or target_per_class <= 0:
        elapsed = time.time() - start_time
        log(INFO, f"[HDH SMOTENC] Skipped: class counts {dict(class_counts)} are too small for SMOTENC.")
        return trainloader, elapsed

    x_compact = _standardized_prostate_to_compact_raw(
        x_standardized, feature_mean, feature_std
    )

    if minority_count < target_per_class:
        k_neighbors = min(5, minority_count - 1)
        sampler = SMOTENC(
            categorical_features=[3],
            sampling_strategy={minority_label: target_per_class},
            random_state=1234,
            k_neighbors=k_neighbors,
        )
        x_resampled, y_resampled = sampler.fit_resample(x_compact, y)
    else:
        x_resampled, y_resampled = x_compact, y

    # Keep all minority observations (original plus synthetic) and deterministically
    # down-sample the majority to the same target. This preserves the original
    # local sample budget used for aggregation and isolates the effect of balance.
    rng = np.random.default_rng(1234)
    selected_indices = []
    for label in sorted(class_counts):
        indices = np.flatnonzero(y_resampled == label)
        if len(indices) > target_per_class:
            indices = rng.choice(indices, size=target_per_class, replace=False)
        selected_indices.extend(indices.tolist())

    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    rng.shuffle(selected_indices)
    x_balanced = x_resampled[selected_indices]
    y_balanced = y_resampled[selected_indices]

    balanced_frame = _compact_raw_to_standardized_prostate(
        x_balanced, y_balanced, feature_mean, feature_std
    )
    balanced_dataset = _frame_to_tensor_dataset(
        balanced_frame, feature_mean, feature_std
    )

    batch_size = trainloader.batch_size or 32
    balanced_loader = DataLoader(
        TensorLabelDataset(balanced_dataset),
        batch_size=batch_size,
        shuffle=True,
    )

    elapsed = time.time() - start_time
    after_counts = Counter(int(v) for v in y_balanced.tolist())
    generated = max(0, target_per_class - minority_count)
    log(
        INFO,
        f"[HDH SMOTENC] Rebalanced prostate data: {dict(class_counts)} -> "
        f"{dict(after_counts)}; generated={generated}; time={elapsed:.4f}s",
    )
    return balanced_loader, elapsed


def rebalance_trainloader_with_hdh(trainloader):
    """Dispatch to the modality-specific HDH without altering existing tasks."""
    if DATASET_NAME == "PROSTATE_CANCER":
        return rebalance_trainloader_with_smotenc(trainloader)
    return rebalance_trainloader_with_gan(trainloader)


def rebalance_trainloader_with_gan(trainloader):
    _t0_hdh = time.time()
    global DATASET_NAME
    if DATASET_NAME == "PROSTATE_CANCER":
        raise NotImplementedError(
            "The added prostate-cancer scale-up task does not define a tabular GAN for "
            "heterogeneous-data-handler. Use this task for the scalability/client-selector "
            "validation unless a tabular augmentation method is added explicitly."
        )
    if DATASET_NAME not in AVAILABLE_DATASETS:
        raise ValueError(f"[ERROR] Dataset '{DATASET_NAME}' non trovato in AVAILABLE_DATASETS.")
    dataset_config = AVAILABLE_DATASETS[DATASET_NAME]
    batch_size = 32
    base = []
    for x, y in trainloader:
        for xi, yi in zip(x, y):
            base.append((xi, yi))
    trainset = balance_dataset_with_gan(
        base,
        num_classes=dataset_config["num_classes"],
        target_per_class=len(base) // dataset_config["num_classes"],
    )
    ds_name = DATASET_NAME.lower()
    if "cifar" in ds_name:
        max_limit = 5000
    elif "imagenet" in ds_name:
        max_limit = 1300
    else:
        max_limit = len(base) // dataset_config["num_classes"]
    trainset = truncate_dataset(trainset, max_limit)
    hdh_ms = (time.time() - _t0_hdh)
    if hdh_ms < 10:
        hdh_ms = 0.0
    log(INFO, f"HDH Data Handler (GAN) Total Processing time: {hdh_ms:.2f} seconds")
    return DataLoader(TensorLabelDataset(trainset), batch_size=batch_size, shuffle=True), hdh_ms


def get_jsd(trainloader):
    log(INFO, "Calculating Jensen-Shannon Divergence (JSD) for dataset distribution...")
    labels = [lbl.item() if isinstance(lbl, torch.Tensor) else lbl for _, lbl in trainloader.dataset]
    dist = dict(Counter(labels))
    num_classes = AVAILABLE_DATASETS[DATASET_NAME]["num_classes"]
    total_samples = sum(dist.values())
    P = np.array([dist.get(i, 0) / total_samples for i in range(num_classes)])
    Q = np.array([1.0 / num_classes] * num_classes)
    M = 0.5 * (P + Q)

    def kl_div(p, q):
        return np.sum([pi * np.log2(pi / qi) if pi > 0 else 0.0 for pi, qi in zip(p, q)])

    JSD = 0.5 * kl_div(P, M) + 0.5 * kl_div(Q, M)
    log(INFO, f"Jensen-Shannon Divergence (client vs perfect IID): {JSD:.2f}")
    return JSD


def train(net, trainloader, valloader, epochs, DEVICE):
    labels = [lbl.item() if isinstance(lbl, torch.Tensor) else lbl for _, lbl in trainloader.dataset]
    dist = dict(Counter(labels))
    log(INFO, f"Training dataset distribution ({DATASET_NAME}): {dist}")
    num_classes = AVAILABLE_DATASETS[DATASET_NAME]["num_classes"]
    log(INFO, "Starting training...")
    start_time = time.time()

    if DATASET_NAME == "PROSTATE_CANCER":
        # The source study fits the same logistic-regression model to convergence
        # in a single FL round. FLiP instead needs meaningful evolution across
        # its 20 rounds, so this task uses incremental full-batch SGD on the same
        # single-logit LR architecture. Each round starts from the current Flower
        # global parameters and performs one local gradient step per configured
        # epoch before FedAvg aggregation.
        net.to(DEVICE)
        net.train()

        xs, ys = [], []
        for batch_x, batch_y in trainloader:
            xs.append(batch_x.to(DEVICE))
            ys.append(batch_y.to(DEVICE))
        X = torch.cat(xs, dim=0).float()
        y = torch.cat(ys, dim=0).float()

        # C=1 in sklearn corresponds to L2 regularization. With a mean-reduced
        # binary cross-entropy objective, 1/(C*N) is the corresponding per-client
        # penalty scale. The defaults below can be overridden at the top level of
        # config.json without affecting any other learning task.
        try:
            with open(config_file, "r") as f:
                prostate_train_cfg = json.load(f)
            prostate_lr = float(prostate_train_cfg.get("prostate_learning_rate", 0.35))
            prostate_c = float(prostate_train_cfg.get("prostate_regularization_c", 1.0))
        except Exception:
            prostate_lr = 0.35
            prostate_c = 1.0
        if prostate_lr <= 0.0:
            raise ValueError("prostate_learning_rate must be > 0.")
        if prostate_c <= 0.0:
            raise ValueError("prostate_regularization_c must be > 0.")

        l2_weight = 1.0 / (prostate_c * max(1, len(trainloader.dataset)))
        criterion = nn.BCEWithLogitsLoss()

        for _ in range(max(1, int(epochs))):
            net.zero_grad(set_to_none=True)
            logits = net(X)
            loss = criterion(logits, y)
            loss = loss + 0.5 * l2_weight * torch.sum(net.linear.weight ** 2)
            loss.backward()
            with torch.no_grad():
                for param in net.parameters():
                    param -= prostate_lr * param.grad

        log(INFO, f"[Prostate LR] Incremental SGD: lr={prostate_lr}, "
                  f"steps={max(1, int(epochs))}, samples={len(trainloader.dataset)}")
    else:
        net.to(DEVICE)
        criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        net.train()
        for _ in range(epochs):
            for images, labels in trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(net(images), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()

    training_time = time.time() - start_time
    log(INFO, f"Training completed in {training_time:.2f} seconds")
    global TRAIN_COMPLETED_TS
    TRAIN_COMPLETED_TS = start_time + training_time
    train_loss, train_acc, train_f1, train_mae = test(net, trainloader)
    val_loss, val_acc, val_f1, val_mae = test(net, valloader)
    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "train_f1": train_f1,
        "train_mae": train_mae,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_f1": val_f1,
        "val_mae": val_mae,
    }
    return results, training_time


def test(net, loader):
    net.to(DEVICE)
    net.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    correct = 0

    if DATASET_NAME == "PROSTATE_CANCER":
        criterion = nn.BCEWithLogitsLoss()
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                logits = net(inputs)
                targets = labels.float()
                loss = criterion(logits, targets)
                total_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).long()
                correct += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    else:
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = net(imgs)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, average='macro')
    try:
        mae = np.mean(np.abs(np.array(all_labels) - np.array(all_preds)))
    except:
        mae = None
    return avg_loss, accuracy, f1, mae


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    state_dict._metadata = {"": {"version": 2}}
    net.load_state_dict(state_dict, strict=True)
