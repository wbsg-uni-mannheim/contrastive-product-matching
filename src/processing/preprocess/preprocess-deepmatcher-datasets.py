import pandas as pd
import numpy as np
np.random.seed(42)
import random
random.seed(42)

import os

def assign_clusterid(identifier, cluster_id_dict, cluster_id_amount):
    try:
        result = cluster_id_dict[identifier]
    except KeyError:
        result = cluster_id_amount
    return result

def preprocess_dataset(arg_tuple):

    handle, id_handle_left, id_handle_right = arg_tuple
    
    print(f'BUILDING {handle} TRAIN, VALID, GS...')
    
    left_df = pd.read_csv(f'../../../data/raw/{handle}/tableA.csv', engine='python')
    right_df = pd.read_csv(f'../../../data/raw/{handle}/tableB.csv', engine='python')
    
    left_df['id'] = f'{id_handle_left}_' +  left_df['id'].astype(str)
    right_df['id'] = f'{id_handle_right}_' +  right_df['id'].astype(str)
    
    left_df = left_df.set_index('id', drop=False)
    right_df = right_df.set_index('id', drop=False)
    left_df = left_df.fillna('')
    right_df = right_df.fillna('')

    train = pd.read_csv(f'../../../data/raw/{handle}/train.csv')
    test = pd.read_csv(f'../../../data/raw/{handle}/test.csv')
    valid = pd.read_csv(f'../../../data/raw/{handle}/valid.csv')
    
    full = train.append(valid, ignore_index=True).append(test, ignore_index=True)
    full = full[full['label'] == 1]
    
    full['ltable_id'] = f'{id_handle_left}_' + full['ltable_id'].astype(str)
    full['rtable_id'] = f'{id_handle_right}_' + full['rtable_id'].astype(str)
    
    bucket_list = []
    for i, row in full.iterrows():
        left = f'{row["ltable_id"]}'
        right = f'{row["rtable_id"]}'
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
    
    cluster_id_dict = {}
    
    for i, id_set in enumerate(bucket_list):
        for v in id_set:
            cluster_id_dict[v] = i
    
    train['ltable_id'] = f'{id_handle_left}_' + train['ltable_id'].astype(str)
    train['rtable_id'] = f'{id_handle_right}_' + train['rtable_id'].astype(str)

    test['ltable_id'] = f'{id_handle_left}_' + test['ltable_id'].astype(str)
    test['rtable_id'] = f'{id_handle_right}_' + test['rtable_id'].astype(str)
                        
    valid['ltable_id'] = f'{id_handle_left}_' + valid['ltable_id'].astype(str)
    valid['rtable_id'] = f'{id_handle_right}_' + valid['rtable_id'].astype(str)

    train['label'] = train['label'].apply(lambda x: int(x))
    test['label'] = test['label'].apply(lambda x: int(x))
    valid['label'] = valid['label'].apply(lambda x: int(x))

    valid['pair_id'] = valid['ltable_id'] + '#' + valid['rtable_id']

    train = train.append(valid, ignore_index=True)

    train_left = left_df.loc[list(train['ltable_id'].values)]
    train_right = right_df.loc[list(train['rtable_id'].values)]
    train_labels = [int(x) for x in list(train['label'].values)]

    gs_left = left_df.loc[list(test['ltable_id'].values)]
    gs_right = right_df.loc[list(test['rtable_id'].values)]
    gs_labels = [int(x) for x in list(test['label'].values)]

    train_left = train_left.reset_index(drop=True)
    train_right = train_right.reset_index(drop=True)
    gs_left = gs_left.reset_index(drop=True)
    gs_right = gs_right.reset_index(drop=True)
    
    cluster_id_amount = len(bucket_list)
    
    train_left['cluster_id'] = train_left['id'].apply(assign_clusterid, args=(cluster_id_dict, cluster_id_amount))
    train_right['cluster_id'] = train_right['id'].apply(assign_clusterid, args=(cluster_id_dict, cluster_id_amount))
    gs_left['cluster_id'] = gs_left['id'].apply(assign_clusterid, args=(cluster_id_dict, cluster_id_amount))
    gs_right['cluster_id'] = gs_right['id'].apply(assign_clusterid, args=(cluster_id_dict, cluster_id_amount))
    
    train_df = train_left.join(train_right, lsuffix='_left', rsuffix='_right')
    train_df['label'] = train_labels
    train_df['pair_id'] = train_df['id_left'] + '#' + train_df['id_right']
    assert len(train_df) == len(train)

    gs_df = gs_left.join(gs_right, lsuffix='_left', rsuffix='_right')
    gs_df['label'] = gs_labels
    gs_df['pair_id'] = gs_df['id_left'] + '#' + gs_df['id_right']
    assert len(gs_df) == len(test)
     
    print(f'Size of training set: {len(train_df)}')
    print(f'Size of gold standard: {len(gs_df)}')
    print(f'Distribution of training set labels: \n{train_df["label"].value_counts()}')
    print(f'Distribution of gold standard labels: \n{gs_df["label"].value_counts()}')
    
    os.makedirs(os.path.dirname(f'../../../data/interim/{handle}/'), exist_ok=True)

    train_df.to_json(f'../../../data/interim/{handle}/{handle}-train.json.gz', compression='gzip', lines=True, orient='records')
    valid['pair_id'].to_csv(f'../../../data/interim/{handle}/{handle}-valid.csv', header=True, index=False)
    gs_df.to_json(f'../../../data/interim/{handle}/{handle}-gs.json.gz', compression='gzip', lines=True, orient='records')

    print(f'FINISHED BULDING {handle} DATASETS\n')

    
if __name__ == '__main__':
    datasets = [
        ('abt-buy', 'abt', 'buy'),
        ('amazon-google', 'amazon', 'google')
    ]
    for dataset in datasets:
        preprocess_dataset(dataset)