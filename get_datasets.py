import bz2
import numpy as np
import pandas as pd
import requests
import os
import shutil
from zipfile import ZipFile
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.datasets import load_iris, load_digits, load_svmlight_file
from sklearn.datasets import fetch_20newsgroups, fetch_openml
from sklearn.preprocessing import MinMaxScaler

def get_dataset(dataset):

    if dataset == 'anuran':
        return get_anuran()
    elif dataset == 'avila':
        return get_avila()
    elif dataset == 'collins':
        return get_collins()
    elif dataset == 'digits':
        return get_digits()
    elif dataset == 'iris':
        return get_iris()
    elif dataset == 'letter':
        return get_letter()
    elif dataset == 'mice':
        return get_mice()
    elif dataset == 'newsgroups':
        return get_newsgroups()
    elif dataset == 'pendigits':
        return get_pendigits()
    elif dataset == 'sensorless':
        return get_sensorless()
    elif dataset == 'vowel':
        return get_vowel()

def get_bz(name, url):
    bzfile = "/tmp/{}.bz".format(name)
    txtfile = "/tmp/{}.txt".format(name)

    r = requests.get(url, allow_redirects=True)

    with open(bzfile, "wb") as f:
        f.write(r.content)
    with open(bzfile, 'rb') as f:
        d = bz2.decompress(f.read())
    with open(txtfile, 'wb') as f:
        f.write(d)

    data = load_svmlight_file(txtfile)[0].toarray()

    os.remove(bzfile)
    os.remove(txtfile)
    return data

def get_anuran():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00406/Anuran%20Calls%20(MFCCs).zip'
    zipresp = requests.get(url, allow_redirects=True)
    with open("/tmp/tempfile.zip", "wb") as f:
        f.write(zipresp.content)
    zf = ZipFile("/tmp/tempfile.zip")
    zf.extractall(path='/tmp/')
    zf.close()
    data = pd.read_csv('/tmp/Frogs_MFCCs.csv').iloc[:,:22].values

    os.remove('/tmp/Frogs_MFCCs.csv')
    return data.astype(float), 10

def get_avila():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00459/avila.zip'
    zipresp = requests.get(url, allow_redirects=True)
    with open("/tmp/tempfile.zip", "wb") as f:
        f.write(zipresp.content)
    zf = ZipFile("/tmp/tempfile.zip")
    zf.extractall(path='/tmp/')
    zf.close()

    with open('/tmp/avila/avila-tr.txt', 'r') as f:
        text = [l for l in f.readlines()]

    data_train = np.array([[float(j) for j in text[i].split(',')[:-1]]
                            for i in range(len(text))])

    with open('/tmp/avila/avila-ts.txt', 'r') as f:
        text = [l for l in f.readlines()]

    data_test = np.array([[float(j) for j in text[i].split(',')[:-1]]
                            for i in range(len(text))])

    data = np.concatenate((data_train, data_test))
    mms = MinMaxScaler()
    data = mms.fit_transform(data)
    shutil.rmtree('/tmp/avila')
    os.remove('/tmp/tempfile.zip')
    return np.array(data), 12

def get_collins():
    url = 'https://www.openml.org/data/get_csv/17953251/php5OMDBD'
    data = pd.read_csv(url).iloc[:,1:-4]
    data = np.array(data)
    mms = MinMaxScaler()
    data = mms.fit_transform(data)
    return data, 30

def get_digits():
    return load_digits().data, 10

def get_iris():
    return load_iris().data, 3

def get_letter():
    data = fetch_openml('letter', version=1).data
    mms = MinMaxScaler()
    return mms.fit_transform(data), 26

def get_mice():
    data = fetch_openml(name="miceprotein", version=4).data
    return data.dropna(axis=0).values, 8

def get_newsgroups():
    dataset = fetch_20newsgroups(subset='all',
                                 remove=('headers', 'footers', 'quotes'),
                                 shuffle=True)
    vectorizer = TfidfVectorizer(stop_words='english',
                                 token_pattern=r'\b[^\d\W]+\b',
                                 min_df=.01, max_df=.1)
    return vectorizer.fit_transform(dataset.data).toarray(), 20

def get_pendigits():
    url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits'
    r = requests.get(url, allow_redirects=True)
    with open('pendigits.txt', 'w') as f:
        f.write(r.text)
    sps = load_svmlight_file('pendigits.txt')
    train = sps[0].toarray()

    url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits.t'
    r = requests.get(url, allow_redirects=True)
    with open('pendigits.txt', 'w') as f:
        f.write(r.text)
    sps = load_svmlight_file('pendigits.txt')
    test = sps[0].toarray()

    data = np.concatenate([train, test])

    os.remove('pendigits.txt')
    return data, 10

def get_sensorless():
    url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/Sensorless.scale'
    r = requests.get(url, allow_redirects=True)
    with open('sensorless.txt', 'w') as f:
        f.write(r.text)
    sps = load_svmlight_file('sensorless.txt')
    data = sps[0].toarray()

    os.remove('sensorless.txt')
    return data, 11

def get_vowel():
    url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vowel.scale'
    r = requests.get(url, allow_redirects=True)
    with open('vowel.txt', 'w') as f:
        f.write(r.text)
    sps = load_svmlight_file('vowel.txt')
    train = sps[0].toarray()

    url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vowel.scale.t'
    r = requests.get(url, allow_redirects=True)
    with open('vowel.txt', 'w') as f:
        f.write(r.text)
    sps = load_svmlight_file('vowel.txt')
    test = sps[0].toarray()

    data = np.concatenate([train, test])

    os.remove('vowel.txt')
    return data, 11