import get_scores as gs
import pandas as pd
import valohai
import os

def retrieve_scores(data, labels, args):
    scores = pd.DataFrame(list(labels.apply(lambda x: gs.get_scores(data, x.values)).values))
    scores['seed'] = labels.columns
    for arg in args:
        scores[arg[0]] = arg[1]
    return scores

def get_scores(dataset, data_path, labels_paths, algo):
    filename = [i for i in os.listdir(data_path) 
                if i.startswith(f'{dataset}_')][0]
    data, k = gs.get_data(data_path, filename)
    label_files = [i for i in os.listdir(labels_paths) 
                    if i.startswith(f'{dataset}_')
                    and i.endswith(f'.csv')]
    all_scores = []
    for l in label_files:
        labels = pd.read_csv(f'{labels_paths}/{l}')
        args = [('dataset', dataset),
                ('algo', algo),
                ('k', k),
                ('n', data.shape[0]),
                ('d', data.shape[1])]
        if algo == 'pr':
            _, alpha, min_dist, use_centroids = l.split('_')
            alpha = float(alpha)
            type_dist = 'min' if min_dist else 'last'
            use_centroids = use_centroids.split('.')[0] == 'True'
            pr_args = [('alpha', alpha), 
                        ('type_dist', type_dist), 
                        ('use_centroids', use_centroids)]
            args.extend(pr_args)
        scores = retrieve_scores(data, labels, args)
        all_scores.append(scores)
    return pd.concat(all_scores)

VH_INPUTS_DIR = os.getenv('VH_INPUTS_DIR', '.inputs')
data_path = f'{VH_INPUTS_DIR}/data_path'
labels_path = f'{VH_INPUTS_DIR}/labels_path'
dataset = valohai.parameters('dataset').value
algo = valohai.parameters('algos').value

print(os.listdir(data_path))
print(os.listdir(labels_path))

if dataset == 'all':
    datasets = [i.split('_')[0] for i in os.listdir(data_path)]
else:
    datasets = [dataset]

skip = ['beer', 'cifar10', 'poker', 'covtype', 'bng']

for d in datasets:
    if d not in skip:
        print(d)
        scores = get_scores(d, data_path, labels_path, 'pr')
        scores_path = valohai.outputs('scores').path(f'{d}_pr.csv')
        scores.to_csv(scores_path, index=False)