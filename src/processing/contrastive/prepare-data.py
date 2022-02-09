import pandas as pd
import numpy as np
np.random.seed(42)
import random
random.seed(42)

from pathlib import Path
import shutil

if __name__ == '__main__':

    categories = ['computers']
    train_sizes = ['small', 'medium', 'large', 'xlarge']

    data = pd.read_pickle('../../../data/interim/wdc-lspc/corpus/preprocessed_english_corpus.pkl.gz')

    relevant_cols = ['id', 'cluster_id', 'brand', 'title', 'description', 'specTableContent']

    for category in categories:
        out_path = f'../../../data/processed/wdc-lspc/contrastive/pre-train/{category}/'
        shutil.rmtree(out_path, ignore_errors=True)
        Path(out_path).mkdir(parents=True, exist_ok=True)
        for train_size in train_sizes:
            ids = pd.read_pickle(f'../../../data/interim/wdc-lspc/training-sets/preprocessed_{category}_train_{train_size}.pkl.gz')
            
            relevant_ids = set()
            relevant_ids.update(ids['id_left'])
            relevant_ids.update(ids['id_right'])
            
            data_selection = data[data['id'].isin(relevant_ids)]
            data_selection = data_selection[relevant_cols]
            data_selection = data_selection.reset_index(drop=True)
            data_selection.to_pickle(f'{out_path}{category}_train_{train_size}.pkl.gz')