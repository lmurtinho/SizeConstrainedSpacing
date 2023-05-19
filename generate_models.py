import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import time

import warnings
warnings.filterwarnings('ignore')

from get_scores import get_scores
from constrained import ConstrainedMaxMST
from feasible import FeasibleSpacing
from sklearn.cluster import KMeans, AgglomerativeClustering
from get_datasets import get_dataset

ROOT = os.getcwd()
MODELS_PATH = f'{ROOT}/results'
RESULTS_PATH = os.getcwd() + '/results'

DATASET_LIST = ['anuran', 'avila', 'collins', 'digits', 
                'iris', 'letter', 'mice', 'newsgroups', 
                'pendigits', 'sensorless', 'vowel']

def get_model(dataset, filepath, single=False):
    df = joblib.load(filepath)
    if single:
        return df[df.dataset==dataset].model.iloc[0]
    else:
        return df[df.dataset==dataset].model

def single_model_result(model, algo, data,
                        min_size=None,
                        sl_model=None,
                        verbose=False):
    start = time.time()
    if algo in ['cm_full', 'cm_fast', 'fs'] and sl_model:
        labels = model.fit_predict(data, sl_model)
    else:
        labels = model.fit_predict(data)
    end = time.time()
    secs = end - start
    if verbose:
        message = f'\tModel {algo} done in {secs:.2f} seconds'
        if algo in ['cm_full', 'cm_fast', 'fs']:
            message += f' for min_size {min_size}'
        print(message)
    res = {'model': model,
           'algorithm': algo,
           'n': len(data),
           'time': secs,
           'smallest_cluster': np.bincount(labels).min(),
           'min_size': min_size,
            }
    return res

def retrieve_results(algo, dataset_list, min_size_factor=None,
                     km_filepath=None, sl_filepath=None, verbose=False):

    results = []

    for dataset in dataset_list:
        
        # Get data and models
        if verbose:
            print(dataset)
        data, k = get_dataset(dataset)
        
        # get min_sizes
        if algo in ['cm_full', 'cm_fast', 'fs'] and km_filepath:
            km_models = get_model(dataset, km_filepath, False)
            min_sizes = km_models.apply(lambda x: np.bincount(x.labels_).min()).values
            min_sizes = (min_sizes // min_size_factor + 1).astype(int)
        else:
            min_sizes = [min_size_factor * len(data) // k]
        
        # get sl_model
        if algo in ['cm_full', 'cm_fast', 'fs'] and sl_filepath:
            sl_model = get_model(dataset, sl_filepath, True)
        else:
            sl_model = None
        
        if algo in ['cm_full', 'cm_fast', 'fs']:
            rng_runs = min_sizes
        elif algo == 'km':
            rng_runs = range(1,11)
        elif algo == 'sl':
            rng_runs = [1]

        if algo == 'cm_full':
            search_all = True
        elif algo == 'cm_fast':
            search_all = False
        else:
            search_all = None
        
        res_dict = {}
        for i in range(len(rng_runs)):
            
            # get min_size and theck if model already ran
            if algo in ['cm_full', 'cm_fast', 'fs']:
                min_size = rng_runs[i]
                if min_size in res_dict:
                    res = res_dict[min_size]
                    res['id'] = i + 1
                    continue
            else:
                min_size = None

            # select model
            if algo in ['cm_full', 'cm_fast']:
                seed = 1
                model = ConstrainedMaxMST(k, min_size, random_state=seed, search_all=search_all)
            if algo=='fs':
                model = FeasibleSpacing(k, min_size)
            if algo=='km':
                seed = rng_runs[i]
                model = KMeans(k, random_state=seed)
            if algo=='sl':
                model = AgglomerativeClustering(k, linkage='single')
            
            # fit model and predict labels          
            res = single_model_result(model, algo, data, min_size,
                                       sl_model, verbose) 
            res['dataset'] = dataset
            res['k'] = k
            res['id'] = i + 1
            score = get_scores(data, res['model'].labels_)
            res = res | score
            if algo == 'km':
                res['seed'] = rng_runs[i]
            if algo in ['cm_full', 'cm_fast']:
                res['seed'] = 1
                res['min_dist_upper_bound'] = model.min_dist_sum
                res['search_all'] = search_all
            if algo in ['cm_full', 'cm_fast', 'fs']:
                res['min_size_factor'] = min_size_factor
                res_dict[min_size] = res
            results.append(res)
    return results


parser = argparse.ArgumentParser()

parser.add_argument('-a', '--algos', default=['sl', 'km', 'fs', 'cm_fast', 'cm_full'],
                    help='algorithms to be used for clustering')
parser.add_argument('-dl', '--dataset_list', nargs='+', default=DATASET_LIST,
                    help="list of datasets for which models will be generated")
parser.add_argument('-hf', '--hour_in_filename', action='store_true',
                    help='adds to the filename the time of file creation')
parser.add_argument('-rp', '--results_path', default=RESULTS_PATH,
                    help="path in which results will be saved; will be created if it doesn't exist")
parser.add_argument('-sp', '--sl_filepath', default=f'{MODELS_PATH}/results_sl.joblib',
                    help="path to file with single-link models")
parser.add_argument('-kp', '--km_filepath', default=f'{MODELS_PATH}/results_km.joblib',
                    help="path to file with k-means models")
parser.add_argument('-ms', '--min_size_factor', type=float, default=0.75,
                    help='for algo in [cm, fs]: size factor of minimum size guarantee')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='prints progress information')
parser.add_argument('-sm', '--save_models', action="store_true",
                    help="saves a .joblib file with the fitted models")
parser.add_argument('-fr', '--full_results', action='store_true',
                     help="saves all results in a single .csv file")

if __name__ == '__main__':
    args = parser.parse_args()
    
    if args.full_results:
        full_results = []

    for algo in args.algos:
    
        # Get results
        if args.verbose:
            print(f'finding models for algorithm {algo}')            
        results = retrieve_results(algo, args.dataset_list, args.min_size_factor, args.km_filepath,
                                   args.sl_filepath, args.verbose)
        results = pd.DataFrame(results)

        if args.full_results:
            full_results.append(results)

        # Save results
        os.makedirs(args.results_path, exist_ok=True)
        if args.hour_in_filename:
            now = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M')
            filepath = f'{args.results_path}/{now}_results_{algo}'
        else:
            filepath = f'{args.results_path}/results_{algo}'
        if args.save_models:
            joblib.dump(results, f'{filepath}.joblib')
        results.drop(columns=['model']).to_csv(f'{filepath}.csv', index=False)
    if args.full_results:
        if args.hour_in_filename:
            now = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M')
            fr_path = f'{args.results_path}/{now}_full_results.csv'
        else:
            fr_path = f'{args.results_path}/full_results.csv'
        pd.concat(full_results).drop(columns=['model']).to_csv(fr_path, index=False)