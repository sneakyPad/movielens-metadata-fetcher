from imdb import IMDb, IMDbDataAccessError
import pandas as pd
import time
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import ast
from collections import defaultdict
import multiprocessing
dct_no_entries = defaultdict(int)
import numpy as np
import itertools
import os
from collections import Counter
from time import sleep
import math
import json
import janitor

def benchmark_string_comparison():
    import time
    dct_foo = {'alpha':3,
               'beta':1,
               'gamma':2}
    a = id('Tomin Hanks')
    b= id('Tomin Cruise')
    avg_a = avg_b= avg_c = avg_d =avg_e= avg_f=0
    for i in range(0,10000):
        start = time.time()
        com = dct_foo['alpha']==dct_foo['beta']
        avg_a +=start - time.time()


        start = time.time()
        com = 3==1
        avg_b += start - time.time()

        start = time.time()
        com = 'alpha' == 'beta'
        avg_c += start - time.time()

        start = time.time()
        com = 'Tomin d. Hanks' == 'Tomin d. Cruise'
        avg_d += start - time.time()

        start = time.time()
        com = id('Tomin Hanks') == id('Tomin Cruise')
        avg_e += start - time.time()

        start = time.time()
        com = a == b
        avg_f += start - time.time()

    print(i)
    print(id('foo'))
    avg_a = (avg_a / i) *1000
    avg_b = (avg_b/i) * 1000
    avg_c = (avg_c/i) * 1000
    avg_d = (avg_d/i) * 1000
    avg_e = (avg_e / i) * 1000
    print(' Avg_a:{} \n Avg_b:{} \n Avg_c:{} \n Avg_d:{} \n Avg_e:{} \n Avg_f:{}'.format(avg_a,avg_b,avg_c,avg_d, avg_e, avg_f ))
# benchmark_string_comparison()
#%%
    import pandas as pd

    # df = df_meta.value_counts()
    # print(df.head())

def my_eval(expression):
    return expression
    #TODO fix this
    try:
        return ast.literal_eval(str(expression))
    except SyntaxError: #e.g. a ":" or "(", which is interpreted by eval as command
            return expression
    except ValueError: #e.g. an entry is nan, in that case just return an empty string
        return ''


def compute_relative_frequency(df_meta):
    print('Compute relative frequency for all columns and their attributes...')
    #Goal is:
    #Cast:
        # Tom Hanks: 0,3%
        # Matt Damon: 0,2%
    # fpp =np.vstack(df_meta['genres'].values)
    # np_array = df_meta['genres'].values
    # tmp_list = []
    # for element in np_array:
    #     tmp_list.extend(eval(element))

    # print(fpp)len_crawled_ids
    #TODO Implement my eval: https://stackoverflow.com/questions/31423864/check-if-string-can-be-evaluated-with-eval-in-python
    dct_rel_freq={}
    for column in tqdm(df_meta.columns, total=len(df_meta.columns)):
        print('Column: {}'.format(column))
    #     ls_ls_casted=[]
    #     for str_elem in df_meta[column].values:
    #         str_elem= str(str_elem)#.replace(':','').replace('(','').replace(')','')
    #         try:
    #             ls_ls_casted.append(ast.literal_eval(str_elem))
    #         except SyntaxError:
    #             ls_ls_casted.append(str_elem)

        ls_ls_casted = [my_eval(str_elem) for str_elem in df_meta[column].values] #cast encoded lists to real list
        # ls_ls_casted = [json.loads(str(str_elem)) for str_elem in df_meta[column].values] #cast encoded lists to real list
        try:
            if(type(ls_ls_casted[0]) == list):
                merged_res = itertools.chain(*ls_ls_casted) #join all lists to one single list
                ls_merged = list(merged_res)
            else:
                ls_merged = ls_ls_casted
            if(column not in ['Unnamed: 0', 'unnamed_0']):
                c = Counter(ls_merged)
                dct_counter = {str(key): value for key, value in c.items()}
                dct_rel_freq[column]={}
                dct_rel_freq[column]['absolute'] = dct_counter
                # print('Column: {}\n\t absolute:{}'.format(dct_rel_freq[column]['absolute']))

                dct_rel_attribute = {str(key): value / sum(c.values()) for key, value in dct_counter.items()} #TODO create a dict with key val
                dct_rel_freq[column]['relative'] = dct_rel_attribute
                # print('\t relative:{}'.format(dct_rel_freq[column]['relative']))

        except TypeError:
            print('TypeError for Column:{} and ls_ls_casted:{} and *ls_ls_casted:{}'.format(column, ls_ls_casted, *ls_ls_casted))


    return dct_rel_freq
    # save_dict_as_json(dct_rel_freq, 'relative_frequency.json')

def save_dict_as_json(dct, name):
    with open('../data/generated/' + name, 'w') as file:

        json.dump(dct, file, indent=4, sort_keys=True)
def load_json_as_dict(name):
    with open('../data/generated/' + name, 'r') as file:
        id2names = json.loads(file)
        return id2names

def load_dataset(small_dataset):
    if (small_dataset):
        print("Load small dataset")
        #%%
        df_movies = pd.read_csv("../data/input/movielens/small/links.csv")
    else:
        print("Load large dataset")
        df_movies = pd.read_csv("../data/input/movielens/large/links.csv")

    return df_movies

#extracts baed on column_name a nested list of the attribute, e.g. cast and creates
# a second list with the respective ids that are looked up in actor2id.
# you can extract more columns by adding them to column_name
def ids2names(df_movies, actor2id, column_name):
    dct_ls_ids = defaultdict(list)
    dct_ls_columns = defaultdict(list)
    for idx, row in tqdm(df_movies.iterrows(), total=df_movies.shape[0]):
        for column in column_name:  # column_name: ['cast','stars']
            if (type(row[column]) == list):
                ls_names = row[column]
            else:
                ls_names = ast.literal_eval(
                    row[column])  # literal_eval casts the list which is encoded as a string to a list

            # ls_names = row[column]
            dct_ls_columns[column] = ls_names
            # dct_ls_columns[column]= dct_ls_columns[column].append(ls_names)

        # if(type(row['cast'])==list):
        #     casts = row['cast']
        # else:
        #     casts = ast.literal_eval(row['cast']) #literal_eval casts the list which is encoded as a string to a list
        # if(type(row['stars'])==list):
        #     stars = row['stars']
        # else:
        #     stars = ast.literal_eval(row['stars'])

        for key, ls_names in dct_ls_columns.items():
            dct_ls_ids[key].append([actor2id[name] for name in dct_ls_columns[key]])
        # ls_ls_cast_ids.append([actor2id[name] for name in casts])
        # ls_ls_stars_ids.append([actor2id[name] for name in stars])

    return dct_ls_columns, dct_ls_ids

def names2ids(df, column_name):
    print('--- Transform names to ids and add an extra column for it ---')
    # df_movies = pd.read_csv("../data/movielens/small/df_movies.csv")
    df_movies = df
    actor2id = defaultdict(lambda: 1+len(actor2id))

    # [ls_casts[0].append(ls) for ls in ls_casts]
    ls_names = []

    #Add all names to one single list
    print('... Collect names:')
    for idx, row in tqdm(df_movies.iterrows(), total=df_movies.shape[0]):
        for column in column_name:
            if(type(row[column])==list):
                ls_names.extend(row[column])
            else:
                ls_names.extend(ast.literal_eval(row[column])) #literal_eval casts the list which is encoded as a string to a list

    # ls_elem = ls_elem.replace("[",'').replace("'",'').split(sep=',')
    c = Counter(ls_names)
    dct_bar = dict(c)
    for elem in list(ls_names):
        actor2id[elem] #Smart because, lambda has everytime a new element was added, a new default value
        # actor2id[elem] = actor2id[elem] + 1 #count the occurence of an actor
        # if (actor2id[elem] == 0): #assign an unique id to an actor/name
        #     actor2id[elem] = len(actor2id)

    print(actor2id)
    id2actor = {value: key for key, value in actor2id.items()}
    utils.save_dict_as_json(actor2id, 'names2ids.json')
    utils.save_dict_as_json(id2actor, 'ids2names.json')
    # print(id2actor[2])

    print("... Assign Ids to names:")
    dct_ls_columns, dct_ls_ids = ids2names(df_movies,actor2id,column_name)

    # lists look like this:
    # dct_ls_columns = {'cast':['wesley snipes','brad pitt'...]
    # dct_ls_ids ={'cast':[22,33,...]}
    for key, ls_names in dct_ls_columns.items():
        df_movies[key+"_id"] = dct_ls_ids[key]

    return df_movies
