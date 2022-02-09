import numpy as np
np.random.seed(42)
import random
random.seed(42)

import pandas as pd

from pathlib import Path
import glob
import gzip
import pickle
from copy import deepcopy

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoConfig

import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
from sklearn.preprocessing import LabelEncoder

from pdb import set_trace

def assign_clusterid(identifier, cluster_id_dict, cluster_id_amount):
    try:
        result = cluster_id_dict[identifier]
    except KeyError:
        result = cluster_id_amount
    return result


# Methods for serializing examples by dataset
def serialize_sample_lspc(sample):
    string = ''
    string = f'{string}[COL] brand [VAL] {" ".join(sample[f"brand"].split(" ")[:5])}'.strip()
    string = f'{string} [COL] title [VAL] {" ".join(sample[f"title"].split(" ")[:50])}'.strip()
    string = f'{string} [COL] description [VAL] {" ".join(sample[f"description"].split(" ")[:100])}'.strip()
    string = f'{string} [COL] specTableContent [VAL] {" ".join(sample[f"specTableContent"].split(" ")[:200])}'.strip()

    return string

def serialize_sample_abtbuy(sample):
    string = ''
    string = f'{string}[COL] brand [VAL] {" ".join(sample[f"brand"].split())}'.strip()
    string = f'{string} [COL] title [VAL] {" ".join(sample[f"name"].split())}'.strip()
    string = f'{string} [COL] price [VAL] {" ".join(str(sample[f"price"]).split())}'.strip()
    string = f'{string} [COL] description [VAL] {" ".join(sample[f"description"].split()[:100])}'.strip()

    return string

def serialize_sample_amazongoogle(sample):
    string = ''
    string = f'{string}[COL] brand [VAL] {" ".join(sample[f"manufacturer"].split())}'.strip()
    string = f'{string} [COL] title [VAL] {" ".join(sample[f"title"].split())}'.strip()
    string = f'{string} [COL] price [VAL] {" ".join(str(sample[f"price"]).split())}'.strip()
    string = f'{string} [COL] description [VAL] {" ".join(sample[f"description"].split()[:100])}'.strip()

    return string

# Class for Data Augmentation
class Augmenter():
    def __init__(self, aug):

        stopwords = ['[COL]', '[VAL]', 'title', 'name', 'description', 'manufacturer', 'brand', 'specTableContent']

        aug_typo = nac.KeyboardAug(stopwords=stopwords, aug_char_p=0.1, aug_word_p=0.1)
        aug_swap = naw.RandomWordAug(action="swap", stopwords=stopwords, aug_p=0.1)
        aug_del = naw.RandomWordAug(action="delete", stopwords=stopwords, aug_p=0.1)
        aug_crop = naw.RandomWordAug(action="crop", stopwords=stopwords, aug_p=0.1)
        aug_sub = naw.RandomWordAug(action="substitute", stopwords=stopwords, aug_p=0.1)
        aug_split = naw.SplitAug(stopwords=stopwords, aug_p=0.1)

        aug = aug.strip('-')

        if aug == 'all':
            self.augs = [aug_typo, aug_swap, aug_split, aug_sub, aug_del, aug_crop, None]
        
        if aug == 'typo':
            self.augs = [aug_typo, None]

        if aug == 'swap':
            self.augs = [aug_swap, None]

        if aug == 'delete':
            self.augs = [aug_del, None]

        if aug == 'crop':
            self.augs = [aug_crop, None]

        if aug == 'substitute':
            self.augs = [aug_sub, None]

        if aug == 'split':
            self.augs = [aug_split, None]

    def apply_aug(self, string):
        aug = random.choice(self.augs)
        if aug is None:
            return string
        else:
            return aug.augment(string)

# Dataset class for general Contrastive Pre-training for WDC computers
class ContrastivePretrainDataset(torch.utils.data.Dataset):
    def __init__(self, path, deduction_set, tokenizer='huawei-noah/TinyBERT_General_4L_312D', max_length=128, intermediate_set=None, clean=False, dataset='lspc', only_interm=False, aug=False):

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=('[COL]', '[VAL]'))
        self.dataset = dataset
        self.aug = aug

        if self.aug:
            self.augmenter = Augmenter(self.aug)

        data = pd.read_pickle(path)

        if dataset == 'abt-buy':
            data['brand'] = ''

        if dataset == 'amazon-google':
            data['description'] = ''
                
        if intermediate_set is not None:
            interm_data = pd.read_pickle(intermediate_set)
            if only_interm:
                data = interm_data
            else:
                data = data.append(interm_data)
        
        data = data.reset_index(drop=True)

        data = data.fillna('')
        data = self._prepare_data(data)

        self.data = data


    def __getitem__(self, idx):
        # for every example in batch, sample one positive from the dataset
        example = self.data.loc[idx].copy()
        selection = self.data[self.data['labels'] == example['labels']]
        # if len(selection) > 1:
        #     selection = selection.drop(idx)
        pos = selection.sample(1).iloc[0].copy()

        # apply augmentation if set
        if self.aug:
            example['features'] = self.augmenter.apply_aug(example['features'])
            pos['features'] = self.augmenter.apply_aug(pos['features'])

        return (example, pos)

    def __len__(self):
        return len(self.data)
    
    def _prepare_data(self, data):

        if self.dataset == 'lspc':
            data['features'] = data.apply(serialize_sample_lspc, axis=1)

        elif self.dataset == 'abt-buy':
            data['features'] = data.apply(serialize_sample_abtbuy, axis=1)

        elif self.dataset == 'amazon-google':
            data['features'] = data.apply(serialize_sample_amazongoogle, axis=1)

        label_enc = LabelEncoder()
        data['labels'] = label_enc.fit_transform(data['cluster_id'])

        self.label_encoder = label_enc

        data = data[['features', 'labels']]

        return data

# Dataset class for Contrastive Pre-training for Abt-Buy and Amazon-Google
# builds correspondence graph from train+val and builds source-aware sampling datasets
# if split=False, corresponds to not using source-aware sampling
class ContrastivePretrainDatasetDeepmatcher(torch.utils.data.Dataset):
    def __init__(self, path, deduction_set, tokenizer='huawei-noah/TinyBERT_General_4L_312D', max_length=128, intermediate_set=None, clean=False, dataset='abt-buy', aug=False, split=True):

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=('[COL]', '[VAL]'))
        self.dataset = dataset
        self.aug = aug

        if self.aug:
            self.augmenter = Augmenter(self.aug)

        data = pd.read_pickle(path)

        if dataset == 'abt-buy':
            data['brand'] = ''

        if dataset == 'amazon-google':
            data['description'] = ''
        
        if clean:
            train_data = pd.read_json(deduction_set, lines=True)
            
            if dataset == 'abt-buy':
                val = pd.read_csv('../../data/interim/abt-buy/abt-buy-valid.csv')
            elif dataset == 'amazon-google':
                val = pd.read_csv('../../data/interim/amazon-google/amazon-google-valid.csv')

            # use 80% of train and val set positives to build correspondence graph
            val_set = train_data[train_data['pair_id'].isin(val['pair_id'])]
            val_set_pos = val_set[val_set['label'] == 1]
            val_set_pos = val_set_pos.sample(frac=0.80)
            val_ids = set()
            val_ids.update(val_set['pair_id'])
            
            train_data = train_data[~train_data['pair_id'].isin(val_ids)]
            train_data = train_data[train_data['label'] == 1]
            train_data = train_data.sample(frac=0.80)

            train_data = train_data.append(val_set_pos)

            # build the connected components by applying binning
            bucket_list = []
            for i, row in train_data.iterrows():
                left = f'{row["id_left"]}'
                right = f'{row["id_right"]}'
                found = False
                for bucket in bucket_list:
                    if left in bucket and row['label'] == 1:
                        bucket.add(right)
                        found = True
                        break
                    elif right in bucket and row['label'] == 1:
                        bucket.add(left)
                        found = True
                        break
                if not found:
                    bucket_list.append(set([left, right]))
            
            cluster_id_amount = len(bucket_list)
            
            #assign labels to connected components and single nodes (at this point single nodes have same label)
            cluster_id_dict = {}
            for i, id_set in enumerate(bucket_list):
                for v in id_set:
                    cluster_id_dict[v] = i
            data = data.set_index('id', drop=False)
            data['cluster_id'] = data['id'].apply(assign_clusterid, args=(cluster_id_dict, cluster_id_amount))
            #data = data[data['cluster_id'] != cluster_id_amount]

            single_entities = data[data['cluster_id'] == cluster_id_amount].copy()

            index = single_entities.index

            if dataset == 'abt-buy':
                left_index = [x for x in index if 'abt' in x]
                right_index = [x for x in index if 'buy' in x]
            elif dataset == 'amazon-google':
                left_index = [x for x in index if 'amazon' in x]
                right_index = [x for x in index if 'google' in x]
            
            # assing increasing integer label to single nodes
            single_entities = single_entities.reset_index(drop=True)
            single_entities['cluster_id'] = single_entities['cluster_id'] + single_entities.index
            single_entities = single_entities.set_index('id', drop=False)
            single_entities_left = single_entities.loc[left_index]
            single_entities_right = single_entities.loc[right_index]
            
            # source aware sampling, build one sample per dataset
            if split:
                data1 = data.copy().drop(single_entities['id'])
                data1 = data1.append(single_entities_left)

                data2 = data.copy().drop(single_entities['id'])
                data2 = data2.append(single_entities_right)

            else:
                data1 = data.copy().drop(single_entities['id'])
                data1 = data1.append(single_entities_left)
                data1 = data1.append(single_entities_right)

                data2 = data.copy().drop(single_entities['id'])
                data2 = data2.append(single_entities_left)
                data2 = data2.append(single_entities_right)

            if intermediate_set is not None:
                interm_data = pd.read_pickle(intermediate_set)
                if dataset != 'lspc':
                    cols = data.columns
                    if 'name' in cols:
                        interm_data = interm_data.rename(columns={'title':'name'})
                    if 'manufacturer' in cols:
                        interm_data = interm_data.rename(columns={'brand':'manufacturer'})
                    interm_data['cluster_id'] = interm_data['cluster_id']+10000

                data1 = data1.append(interm_data)
                data2 = data2.append(interm_data)

            data1 = data1.reset_index(drop=True)
            data2 = data2.reset_index(drop=True)

            label_enc = LabelEncoder()
            cluster_id_set = set()
            cluster_id_set.update(data1['cluster_id'])
            cluster_id_set.update(data2['cluster_id'])
            label_enc.fit(list(cluster_id_set))
            data1['labels'] = label_enc.transform(data1['cluster_id'])
            data2['labels'] = label_enc.transform(data2['cluster_id'])

            self.label_encoder = label_enc
                
        data1 = data1.reset_index(drop=True)

        data1 = data1.fillna('')
        data1 = self._prepare_data(data1)

        data2 = data2.reset_index(drop=True)

        data2 = data2.fillna('')
        data2 = self._prepare_data(data2)

        diff = abs(len(data1)-len(data2))

        if len(data1) > len(data2):
            if len(data2) < diff:
                sample = data2.sample(diff, replace=True)
            else:
                sample = data2.sample(diff)
            data2 = data2.append(sample)
            data2 = data2.reset_index(drop=True)

        elif len(data2) > len(data1):
            if len(data1) < diff:
                sample = data1.sample(diff, replace=True)
            else:
                sample = data1.sample(diff)
            data1 = data1.append(sample)
            data1 = data1.reset_index(drop=True)

        self.data1 = data1
        self.data2 = data2

    def __getitem__(self, idx):
        # for every example, sample one positive from the respective sampling dataset
        example1 = self.data1.loc[idx].copy()
        selection1 = self.data1[self.data1['labels'] == example1['labels']]
        # if len(selection1) > 1:
        #     selection1 = selection1.drop(idx)
        pos1 = selection1.sample(1).iloc[0].copy()

        example2 = self.data2.loc[idx].copy()
        selection2 = self.data2[self.data2['labels'] == example2['labels']]
        # if len(selection2) > 1:
        #     selection2 = selection2.drop(idx)
        pos2 = selection2.sample(1).iloc[0].copy()

        # apply augmentation if set
        if self.aug:
            example1['features'] = self.augmenter.apply_aug(example1['features'])
            pos1['features'] = self.augmenter.apply_aug(pos1['features'])
            example2['features'] = self.augmenter.apply_aug(example2['features'])
            pos2['features'] = self.augmenter.apply_aug(pos2['features'])

        return ((example1, pos1), (example2, pos2))

    def __len__(self):
        return len(self.data1)
    
    def _prepare_data(self, data):

        if self.dataset == 'lspc':
            data['features'] = data.apply(serialize_sample_lspc, axis=1)

        elif self.dataset == 'abt-buy':
            data['features'] = data.apply(serialize_sample_abtbuy, axis=1)

        elif self.dataset == 'amazon-google':
            data['features'] = data.apply(serialize_sample_amazongoogle, axis=1)

        data = data[['features', 'labels']]

        return data

# Dataset class for pair-wise cross-entropy fine-tuning
class ContrastiveClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, path, dataset_type, size=None, tokenizer='huawei-noah/TinyBERT_General_4L_312D', max_length=128, dataset='lspc', aug=False):

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=('[COL]', '[VAL]'))
        self.dataset_type = dataset_type
        self.dataset = dataset
        self.aug = aug

        if self.aug:
            self.augmenter = Augmenter(self.aug)

        if dataset == 'lspc':
            data = pd.read_pickle(path)
        else:
            data = pd.read_json(path, lines=True)
        
        if dataset == 'abt-buy':
            data['brand_left'] = ''
            data['brand_right'] = ''

        if dataset == 'amazon-google':
            data['description_left'] = ''
            data['description_right'] = ''

        data = data.fillna('')

        if self.dataset_type != 'test':
            if dataset == 'lspc':
                validation_ids = pd.read_csv(f'../../data/raw/wdc-lspc/validation-sets/computers_valid_{size}.csv')
            elif dataset == 'abt-buy':
                validation_ids = pd.read_csv(f'../../data/interim/abt-buy/abt-buy-valid.csv')
            elif dataset == 'amazon-google':
                validation_ids = pd.read_csv(f'../../data/interim/amazon-google/amazon-google-valid.csv')
            if self.dataset_type == 'train':
                data = data[~data['pair_id'].isin(validation_ids['pair_id'])]
            else:
                data = data[data['pair_id'].isin(validation_ids['pair_id'])]

        data = data.reset_index(drop=True)

        data = self._prepare_data(data)

        self.data = data


    def __getitem__(self, idx):
        example = self.data.loc[idx].copy()

        if self.aug:
            example['features_left'] = self.augmenter.apply_aug(example['features_left'])
            example['features_right'] = self.augmenter.apply_aug(example['features_right'])

        return example

    def __len__(self):
        return len(self.data)
    
    def _prepare_data(self, data):

        if self.dataset == 'lspc':
            data['features_left'] = data.apply(self.serialize_sample_lspc, args=('left',), axis=1)
            data['features_right'] = data.apply(self.serialize_sample_lspc, args=('right',), axis=1)
        elif self.dataset == 'abt-buy':
            data['features_left'] = data.apply(self.serialize_sample_abtbuy, args=('left',), axis=1)
            data['features_right'] = data.apply(self.serialize_sample_abtbuy, args=('right',), axis=1)
        elif self.dataset == 'amazon-google':
            data['features_left'] = data.apply(self.serialize_sample_amazongoogle, args=('left',), axis=1)
            data['features_right'] = data.apply(self.serialize_sample_amazongoogle, args=('right',), axis=1)

        data = data[['features_left', 'features_right', 'label']]
        data = data.rename(columns={'label': 'labels'})

        return data

    def serialize_sample_lspc(self, sample, side):
        
        string = ''
        string = f'{string}[COL] brand [VAL] {" ".join(sample[f"brand_{side}"].split(" ")[:5])}'.strip()
        string = f'{string} [COL] title [VAL] {" ".join(sample[f"title_{side}"].split(" ")[:50])}'.strip()
        string = f'{string} [COL] description [VAL] {" ".join(sample[f"description_{side}"].split(" ")[:100])}'.strip()
        string = f'{string} [COL] specTableContent [VAL] {" ".join(sample[f"specTableContent_{side}"].split(" ")[:200])}'.strip()

        return string

    def serialize_sample_abtbuy(self, sample, side):
        
        string = ''
        string = f'{string}[COL] brand [VAL] {" ".join(sample[f"brand_{side}"].split())}'.strip()
        string = f'{string} [COL] title [VAL] {" ".join(sample[f"name_{side}"].split())}'.strip()
        string = f'{string} [COL] price [VAL] {" ".join(str(sample[f"price_{side}"]).split())}'.strip()
        string = f'{string} [COL] description [VAL] {" ".join(sample[f"description_{side}"].split()[:100])}'.strip()
        

        return string

    def serialize_sample_amazongoogle(self, sample, side):
        
        string = ''
        string = f'{string}[COL] brand [VAL] {" ".join(sample[f"manufacturer_{side}"].split())}'.strip()
        string = f'{string} [COL] title [VAL] {" ".join(sample[f"title_{side}"].split())}'.strip()
        string = f'{string} [COL] price [VAL] {" ".join(str(sample[f"price_{side}"]).split())}'.strip()
        string = f'{string} [COL] description [VAL] {" ".join(sample[f"description_{side}"].split()[:100])}'.strip()

        return string