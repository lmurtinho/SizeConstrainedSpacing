import joblib

def get_model(dataset, filepath, single=False):
    df = joblib.load(filepath)
    if single:
        return df[df.dataset==dataset].model.iloc[0]
    else:
        return df[df.dataset==dataset].model

def single_model_result(model, algo, data, min_size=None, 
                        sl_model=None, verbose=False):
    start = time.time()
    if algo in ['cm', 'fs'] and sl_filepath:
        labels = model.fit_predict(data, sl_model)
    else:
        labels = model.fit_predict(data)
    end = time.time()
    secs = end - start
    if verbose:
        message = f'\tModel {algo} done in {secs:.2f} seconds'
        if algo in ['cm', 'fs']:
            message += f' for min_size {min_size}'
        print(message)
    res = {'model': model,
           'dataset': dataset,
           'algorithm': algo,
           'n': len(data),
           'k': k,
           'id': i + 1,
           'seed': seed,
           'time': secs,
           'smallest_cluster': np.bincount(labels).min(),
           'min_size': min_size,
            }
    return res