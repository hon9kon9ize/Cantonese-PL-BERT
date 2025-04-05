# coding: utf-8

import os
import os.path as osp
import time
import random
import numpy as np
import random

import string
import pickle

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from text_utils import TextCleaner

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

np.random.seed(1)
random.seed(1)


class FilePathDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        tokenizer="hon9kon9ize/bert-large-cantonese",
        word_separator=102,  # [SEP]
        token_separator="[SEP]",
        token_mask="[MASK]",
        max_mel_length=512,
        word_mask_prob=0.15,
        phoneme_mask_prob=0.1,
        replace_prob=0.2,
    ):

        self.data = dataset
        self.max_mel_length = max_mel_length
        self.word_mask_prob = word_mask_prob
        self.phoneme_mask_prob = phoneme_mask_prob
        self.replace_prob = replace_prob
        self.text_cleaner = TextCleaner()
        self.word_separator = word_separator
        self.token_mask = token_mask
        self.token_separator = token_separator

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        phonemes = self.data[idx]["phonemes"]
        input_ids = self.data[idx]["input_ids"]

        token_separator_id = self.text_cleaner.encode(self.token_separator)
        words = []
        phoneme = []
        phonemes, word2ph = self.text_cleaner(" ".join(phonemes))
        labels = []
        masked_index = []
        start_idx = 0

        for i, input_id in enumerate(input_ids):
            words.extend([input_id] * word2ph[i])
            words.append(self.word_separator)
            labels.extend(
                [phonemes[start_idx + j] for j in range(word2ph[i])]
                + [token_separator_id]
            )

            if np.random.rand() < self.word_mask_prob:
                if np.random.rand() < self.replace_prob:
                    if np.random.rand() < (self.phoneme_mask_prob / self.replace_prob):
                        for j in range(word2ph[i]):
                            phoneme.append(
                                phonemes[np.random.randint(0, len(phonemes))]
                            )  # randomized
                    else:
                        phoneme.extend(phonemes[start_idx : start_idx + word2ph[i]])
                else:
                    phoneme.extend(
                        [self.text_cleaner.encode(self.token_mask)] * word2ph[i]
                    )

                masked_index.extend(
                    (np.arange(len(phoneme) - word2ph[i], len(phoneme))).tolist()
                )
            else:
                phoneme.extend(phonemes[start_idx : start_idx + word2ph[i]])

            start_idx += word2ph[i]
            phoneme.append(token_separator_id)

        mel_length = len(phoneme)
        masked_idx = np.array(masked_index)
        masked_index = []
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            phoneme = phoneme[random_start : random_start + self.max_mel_length]
            words = words[random_start : random_start + self.max_mel_length]
            labels = labels[random_start : random_start + self.max_mel_length]

            for m in masked_idx:
                if m >= random_start and m < random_start + self.max_mel_length:
                    masked_index.append(m - random_start)
        else:
            masked_index = masked_idx

        assert len(phoneme) == len(words), f"{len(phoneme)} != {len(words)}"
        assert len(phoneme) == len(labels), f"{len(phoneme)} != {len(labels)}"

        phonemes = torch.LongTensor(phoneme)
        labels = torch.LongTensor(labels)
        words = torch.LongTensor(words)

        return phonemes, words, labels, masked_index


class Collator(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.return_wave = return_wave

    def __call__(self, batch):
        # batch[0] = wave, mel, text, f0, speakerid
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[1].shape[0] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        max_text_length = max([b[1].shape[0] for b in batch])

        words = torch.zeros((batch_size, max_text_length)).long()
        labels = torch.zeros((batch_size, max_text_length)).long()
        phonemes = torch.zeros((batch_size, max_text_length)).long()
        input_lengths = []
        masked_indices = []
        for bid, (phoneme, word, label, masked_index) in enumerate(batch):
            text_size = phoneme.size(0)
            words[bid, :text_size] = word
            labels[bid, :text_size] = label
            phonemes[bid, :text_size] = phoneme
            input_lengths.append(text_size)
            masked_indices.append(masked_index)
        output_dict = {
            "phonemes": phonemes,
            "words": words,
            "labels": labels,
            "input_lengths": input_lengths,
            "masked_indices": masked_indices,
        }

        return output_dict


def build_dataloader(
    df,
    validation=False,
    batch_size=4,
    num_workers=1,
    device="cpu",
    collate_config={},
    dataset_config={},
):

    dataset = FilePathDataset(df, **dataset_config)
    collate_fn = Collator(**collate_config)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(not validation),
        num_workers=num_workers,
        drop_last=(not validation),
        collate_fn=collate_fn,
        pin_memory=(device != "cpu"),
    )

    return data_loader
