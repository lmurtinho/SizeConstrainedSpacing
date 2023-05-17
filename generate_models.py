import os
import joblib
import numpy as np
import pandas as pd
import time
from datetime import datetime
import argparse

import warnings
warnings.filterwarnings('ignore')

from get_scores import get_scores
from helper import get_model, single_model_result
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

def retrieve_results(algo, dataset_list, min_size_factor, km_filepath, sl_filepath,
                     seed=None, search_all=True, verbose=False):

    results = []

    for dataset in dataset_list:
        
        # Get data and models
        if verbose:
            print(dataset)
        data, k = get_dataset(dataset)
        
        # get min_sizes
        if args.algo in ['cm', 'fs'] and km_filepath:
            km_models = get_model(dataset, km_filepath, False)
            min_sizes = km_models.apply(lambda x: np.bincount(x.labels_).min()).values
            min_sizes = (min_sizes / min_size_factor + 1).astype(int)
        else:
            min_sizes = [min_size_factor * len(data) / k]
        
        # get sl_model
        if args.algo in ['cm', 'fs'] and sl_filepath:
            sl_model = get_model(dataset, sl_filepath, True)
        else:
            sl_model = None
        
        if algo in ['cm', 'fs']:
            rng_runs = min_sizes
        elif algo == 'km':
            rng_runs = range(1,11)
        elif algo == 'sl':
            rng_runs == [1]
        
        res_dict = {}
        for i in range(len(rng_runs)):
            
            # get min_size and theck if model already ran
            if algo in ['cm', 'fs']:
                min_size = rng_runs[i]
                if min_size in res_dict:
                    res = res_dict[min_size]
                    res['id'] = i + 1
                    continue
            else:
                min_size = None

            # select model
            if algo=='cm':
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
            res = single_model_results(algo, data, min_size, 
                                       sl_model, verbose) 
            score = get_scores(data, res['model'].labels_)
            res = res | score
            if algo == 'km':
                res['seed'] = rng_runs[i]
            if algo == 'cm':
                res['seed'] = 1
                res['min_dist_upper_bound'] = model.min_dist_sum,
                res['search_all'] = search_all
            if algo in ['cm', 'fs']:
                res['min_size_factor'] = min_size_factor
                res_dict[min_size] = res
            results.append(res)
    return results


parser = argparse.ArgumentParser()

parser.add_argument('-a', '--algo', default='cm', choices=['cm', 'fs', 'km', 'sl'],
                    help='algorithm to be used for clustering')
parser.add_argument('-dl', '--dataset_list', nargs='+', default=DATASET_LIST,
                    help="list of datasets for which models will be generated")
parser.add_argument('-hf', '--hour_in_filename', action='store_true',
                    help='adds to the filename the time of file creation')
parser.add_argument('-rp', '--results_path', default=RESULTS_PATH,
                    help="path in which results will be saved; will be created if it doesn't exist")
parser.add_argument('-sp', '--sl_filepath', default=f'{MODELS_PATH}/results_SL.joblib',
                    help="path to file with single-link models")
parser.add_argument('-kp', '--km_filepath', default=f'{MODELS_PATH}/results_KM.joblib',
                    help="path to file with k-means models")
parser.add_argument('-ms', '--min_size_factor', type=float, default=0.75,
                    help='for algo in [cm, fs]: size factor of minimum size guarantee')
parser.add_argument('-sa', '--search_all', action='store_true',
                    help="for algo = cm: searches through all possible partitions from 2 to k for each model")
parser.add_argument('-v', '--verbose', action='store_true',
                    help='prints progress information')

if __name__ == '__main__':
    args = parser.parse_args()
    
    # Get results
    if args.verbose:
        print('finding models for algorithm', args.algo)
        if args.algo == 'cm':
            print('search_all:', args.search_all)
    results = retrieve_results(args.algo, args.dataset_list, args.min_size_factor, args.km_filepath, 
                               args.sl_filepath, args.search_all, args.verbose)
    full_results = pd.DataFrame(results)
    
    # Save results
    os.makedirs(args.results_path, exist_ok=True)
    if args.hour_in_filename:
        now = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M')
        filename = f'{args.results_path}/{now}_results_{args.algo}'
    else:
        filename = f'{args.results_path}/results_{args.algo}'
    if args.algo == 'cm':
        if args.search_all:
            filename += '_exhaustive'
        else:
            filename += '_fast'
    joblib.dump(full_results, f'{filename}.joblib')
    full_results.drop(columns=['model']).to_csv(f'{filename}.csv', index=False)