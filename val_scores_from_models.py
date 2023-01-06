import get_scores as gs
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import valohai
import joblib
import os
import json
import pandas as pd

def get_uv_dists(model, data):
    u_centers = model.u_centers
    v_centers = model.v_centers
    dists = np.diag(euclidean_distances(data[u_centers], 
                                        data[v_centers]))
    return dists

def get_scores(model, data, algo):
    scores = gs.get_scores(data, model.labels_)
    scores['n_per_cluster'] = np.bincount(model.labels_)
    if algo == 'pr':
        scores['uv_dists'] = get_uv_dists(model, data)
    else:
        scores['uv_dists'] = []
    return scores

VH_INPUTS_DIR = os.getenv('VH_INPUTS_DIR', '.inputs')
data_path = f'{VH_INPUTS_DIR}/data_path'
models_path = f'{VH_INPUTS_DIR}/models_path'
dataset = valohai.parameters('dataset').value
algo = valohai.parameters('algo').value

# data_path = '/home/ubuntu/PRClustering/csv'
# models_path = '/home/ubuntu/PRClustering/models'
# dataset = '["iris"]'
# algo = 'all'

if dataset == 'all':
    datasets = [i.split('_')[0] for i in os.listdir(data_path)]
else:
    datasets = json.loads(dataset)

if algo == 'all':
    algos = ['pr', 'km', 'sl']
else:
    algos = json.loads(algo)

model_files = os.listdir(models_path)
data_files = os.listdir(data_path)

for d in datasets:
    print(d)
    files = [f for f in data_files if f.startswith(f'{d}_')]
    print(files)
    if files:
        data, k = gs.get_data(data_path, files[0])
        for a in algos:
            if (f'{d}_{a}_models.pkl' in model_files):
                models = joblib.load(f'{models_path}/{d}_{a}_models.pkl')
                dfm = pd.DataFrame(models)
                scores = dfm.model.apply(lambda x: get_scores(x, data, a))
                scores = pd.DataFrame(scores.tolist())
                fdf = pd.concat([dfm, scores], axis=1)
                scores_path = valohai.outputs('scores').path(f'{d}_{a}.pkl')
                # scores_path = f'/home/ubuntu/PRClustering/nscores/{d}_{a}.pkl'
                joblib.dump(fdf, scores_path)