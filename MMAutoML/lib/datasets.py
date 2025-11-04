import os
import cv2
import json
import torch
import scipy
import scipy.io as sio
from skimage import io

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import contextlib
from collections import Counter

import numpy as np
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from transformers import BertTokenizer

Image.MAX_IMAGE_PIXELS = 200_000_000

class Vocab(object):
    def __init__(self, emptyInit=False):
        if emptyInit:
            self.stoi, self.itos, self.vocab_sz = {}, [], 0
        else:
            self.stoi = {
                w: i
                for i, w in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
            }
            self.itos = [w for w in self.stoi]
            self.vocab_sz = len(self.itos)

    def add(self, words):
        cnt = len(self.itos)
        for w in words:
            if w in self.stoi:
                continue
            self.stoi[w] = cnt
            self.itos.append(w)
            cnt += 1
        self.vocab_sz = len(self.itos)

def get_labels_and_frequencies(data):
    label_freqs = Counter()
    data_labels = [item["label"] for item in data]
    if type(data_labels[0]) == list:
        for label_row in data_labels:
            label_freqs.update(label_row)
    else:
        label_freqs.update(data_labels)

    return list(label_freqs.keys()), label_freqs

def get_vocab(args):
    vocab = Vocab()
    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
    
    vocab.stoi = bert_tokenizer.vocab
    vocab.itos = bert_tokenizer.ids_to_tokens
    vocab.vocab_sz = len(vocab.itos)

    return vocab

def format_mmimdb_dataset(data_path):
    train_label_set = set()
    is_save_sample = True
    all_data = {}

    with open(os.path.join(data_path, "mmimdb/split.json")) as fin:
        data_splits = json.load(fin)

    for split_name in data_splits:
        all_data[split_name] = []

        for idx in data_splits[split_name]:
            with open(os.path.join(data_path, "mmimdb/dataset/{}.json".format(idx))) as fin:
                data = json.load(fin)

            plot_id = np.array([len(p) for p in data["plot"]]).argmax()
            dobj = {
                "id": idx,
                "text": data["plot"][plot_id],
                "image": "mmimdb/dataset/{}.jpeg".format(idx),
                "label": data["genres"],
            }

            if any(l in dobj["label"] for l in ["News", "Adult", "Talk-Show", "Reality-TV"]):
                continue

            if split_name == "train":
                train_label_set.update(dobj["label"])
            else:
                for label in dobj["label"]:
                    if label not in train_label_set:
                        is_save_sample = False

            if len(dobj["text"]) > 0 and is_save_sample:
                all_data[split_name].append(dobj)

            is_save_sample = True

    return all_data


class MMIMDBDataset(Dataset):
    def __init__(self, data_path, tokenizer, transform, vocab, args, folder_name):
        self.data = args.all_data[folder_name]
        self.data_dir = data_path
        self.tokenizer = tokenizer
        self.args = args
        self.vocab = vocab
        self.n_classes = len(args.labels)
        self.max_seq_len = args.max_seq_len
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = (self.tokenizer(self.data[index]["text"])[: (self.args.max_seq_len - 1)])
        segment = torch.zeros(len(sentence))

        sentence = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence
            ]
        )

        if self.args.task_type == "multilabel":
            label = torch.zeros(self.n_classes)
            label[
                [self.args.labels.index(tgt) for tgt in self.data[index]["label"]]
            ] = 1
        else:
            label = torch.LongTensor(
                [self.args.labels.index(self.data[index]["label"][0])]
            )

        image = Image.open(os.path.join(self.data_dir, self.data[index]["image"])).convert("RGB")

        image = self.transform(image)

        return sentence, segment, image, label

def build_dataset(is_train, args, folder_name=None):
    transform = build_transform(is_train, args)

    if args.dataset_name == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 10
    elif args.dataset_name.upper() =='MMIMDB':
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize

        args.all_data = format_mmimdb_dataset(args.data_path)

        args.labels, args.label_freqs = get_labels_and_frequencies(args.all_data["train"])
        vocab = get_vocab(args)
        args.vocab = vocab
        args.vocab_sz = vocab.vocab_sz

        dataset = MMIMDBDataset(args.data_path, tokenizer, transform, vocab, args, folder_name)
        nb_classes = len(args.labels)

    return dataset, nb_classes

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def collate_fn(batch, args):
    lens = [len(row[0]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

    text_tensor = torch.zeros(bsz, max_seq_len).long()
    segment_tensor = torch.zeros(bsz, max_seq_len).long()

    img_tensor = torch.stack([row[2] for row in batch])

    if args.task_type == "multilabel":
        # Multilabel case
        tgt_tensor = torch.stack([row[3] for row in batch])
    else:
        # Single Label case
        tgt_tensor = torch.cat([row[3] for row in batch]).long()

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        tokens, segment = input_row[:2]
        text_tensor[i_batch, :length] = tokens
        segment_tensor[i_batch, :length] = segment

    return (text_tensor, segment_tensor, img_tensor), tgt_tensor