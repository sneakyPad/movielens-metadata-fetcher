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
from utils import plot_utils, utils
import random
# manager = multiprocessing.Manager()
# shared_list = manager.list()


def fetch_example():
    # create an instance of the IMDb class
    ia = IMDb()

    # get a movie
    movie = ia.get_movie('0133093')
    print(ia.get_movie_keywords('0133093'))



    print('movie \n{}'.format(movie))
    # print the names of the directors of the movie
    print('Directors:')
    for director in movie['directors']:
        print(director['name'])

    # print the genres of the movie
    print('Genres:')
    for genre in movie['genres']:
        print(genre)

    # search for a person name
    people = ia.search_person('Mel Gibson')
    for person in people:
       print(person.personID, person['name'])


def beautify_names(dct_data, key):
    # clean actors:
    # start_time = time.time()
    ls_names = []
    try:
        for actor in dct_data[key]:
            if(bool(actor)):
                ls_names.append(actor['name'])
    except KeyError:
        dct_no_entries[key]+=1
        # print()No entries for key:

    # print("--- %s seconds ---" % (time.time() - start_time))
    # total_time_one +=time.time() - start_time
    return ls_names

def remove_keys(dict, keys):
    if(keys == None):
        keys = ['certificates', 'cover url', 'thanks',
                      'special effects companies', 'transportation department',
                      'make up department', 'special effects', 'stunts', 'costume departmen',
                      'location management', 'editorial department', 'casting directors', 'art directors',
                      'production managers', 'art department', 'sound department',
                      'visual effects', 'camera department', 'costume designers'
                      'casting department', 'miscellaneous', 'akas', 'production companies', 'distributors',
                      'other companies', 'synopsis', 'cinematographers', 'production designers',
                      'custom designers', 'Opening Weekend United Kingdom', 'Opening Weekend United States']
    for key in keys:
        dict.pop(key, None)
    return dict

def fetch_movie(id, imdb):
    # TODO Actually it should be checked whether this is single process or not, bc the IMDB Peer error occurs only w/ multiprocessing
    movie = imdb.get_movie(id)

    # TODO Optional: select metadata
    dct_data = movie.data

    # to be cleaned:
    keys_to_beautify = ['cast', 'directors', 'writers', 'producers', 'composers', 'editors',
                        'animation department', 'casting department', 'music department', 'set decorators',
                        'script department', 'assistant directors', 'writer', 'director', 'costume designers']
    for key in keys_to_beautify:
        dct_data[key] = beautify_names(dct_data, key)

    # unwrap box office:
    try:
        dct_data.update(dct_data['box office'])
        del dct_data['box office']
    except KeyError:
        pass
        # print('Unwrap: key error for movieId:{} '.format(movie.movieID))# dct_data['title']

    dct_data = remove_keys(dct_data, None)
    return dct_data


def fetch_by_imdb_ids(ls_tpl_ids):
    imdb = IMDb()
    ls_metadata =[]

    # cnt_connection_reset=0
    try:
        # Example:
        # (103,1) => entire movie + metadata is missing
        # (103,0) => only metadata is missing
        for tpl_id_missing in tqdm(ls_tpl_ids, total = len(ls_tpl_ids)):     # loop through ls_ids
                dct_data={}

                id=tpl_id_missing[0]
                is_movie_missing = tpl_id_missing[1]

                tt_id = imdb_id_2_full_Id(id)

                sleep_t = random.randint(2,7)
                sleep(sleep_t)  # Time in seconds

                # if(crawl_from_scratch[0][0]):
                dct_data['imdbId'] = id

                if(is_movie_missing):
                    dct_data = fetch_movie(id, imdb)

                #Fetch stars of the movie with bs4
                ls_stars = fetch_stars(tt_id)
                dct_data['stars'] =ls_stars

                #add dict to the list of all metadata
                ls_metadata.append(dct_data)
    except Exception:
        print('Exception for id:{}'.format(id))
        # cnt_connection_reset+=1
    return ls_metadata, dct_no_entries



def fetch_stars(id):

    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '3600',
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
    }
    ls_stars = []

    try:
        url = "https://www.imdb.com/title/{}/?ref_=rvi_tt".format(id)
        req = requests.get(url, headers)
        soup = BeautifulSoup(req.content, 'html.parser')
        h4_stars = soup.find("h4", text='Stars:')
        div_tag = h4_stars.parent
        next_a_tag = div_tag.findNext('a')
        while (next_a_tag.name != 'span'):
            if (next_a_tag.name == 'a'):
                ls_stars.append(str(next_a_tag.contents[0]))#str() casts from NavigabelString to string
            next_a_tag = next_a_tag.next_sibling
            # class 'bs4.element.Tag'>
        # next_a_tag.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling['class'][0] == 'ghost'
        # print(ls_stars)
    except AttributeError:
        print('AttributeError (most likely no stars are available), movieId:{}'.format(id))
    finally:
        return ls_stars

# TODO Unfinished: This is not done yet
def enhance_by_stars(df):
    pass
    # tqdm.pandas(desc="my bar!")
    # df['stars'] = df['movieID'].apply(lambda id: fetch_stars(id))
    # return df

def print_exception_statistic(dct_no_entries, len_crawled_ids):
    print('[--------- Exception Statistics ---------]')
    # print('No. of ConnectionResetError: {}'.format(cnt_reset))
    print('Joined No-Keys-Exception for the following keys:')
    for key, value in dct_no_entries.items():
        print("\tKey: {}, count:{}, relative: {}".format(key, value, value /len_crawled_ids))
    print('[----------------------------------------]')


def worker(ids):#ids, crawl_from_scratch
    # ids = args[0]
    # ls_missing_imdb_ids = args[1]
    # global shared_list https://stackoverflow.com/questions/40630428/share-a-list-between-different-processes-in-python
    metadata, dct_no_entries = fetch_by_imdb_ids(ids)
    # shared_list.extend(metadata)
    # print('worker done')
    return metadata, dct_no_entries

def crawl_metadata(ls_imdb_ids,multi_processing, no_processes, develop_size):

    print('Fetching metadata of {} movies'.format(len(ls_imdb_ids)))
    if(develop_size>0):
        ls_imdb_ids = ls_imdb_ids[:develop_size]

    if (multi_processing):
        print('Start multiprocessing...')
        start_time = time.time()  # measure time

        no_processes = no_processes
        ls_ls_metadata = []
        ls_dct_exceptions = []
        cnt_reset = 0

        len_dataset = len(ls_imdb_ids)
        ls_splitted = np.array_split(np.array(ls_imdb_ids), no_processes)
        # ls_missing_imdb_ids = np.array_split(np.array(ls_missing_imdb_ids), no_processes)

        pool = multiprocessing.Pool(processes=no_processes)
        # m = multiprocessing.Manager()
        # q = m.Queue()

        # Pool.map returns list of pairs: https://stackoverflow.com/questions/39303117/valueerror-too-many-values-to-unpack-multiprocessing-pool
        for ls_metadata, dct_no_entries in pool.map(worker,ls_splitted):  # ls_ls_metadata=pool.map(worker, ls_splitted):
            # append both objects to a separate list
            ls_ls_metadata.append(ls_metadata)
            ls_dct_exceptions.append(dct_no_entries)


        print("--- %s seconds ---" % (time.time() - start_time))

        merged_res = itertools.chain(*ls_ls_metadata)  # unpack the list to merge n lists
        ls_metadata = list(merged_res)

        df_exceptions = pd.DataFrame(ls_dct_exceptions).sum()  # sum over all rows

        print_exception_statistic(df_exceptions.to_dict(),len(ls_imdb_ids))
        print("--- %s seconds ---" % (time.time() - start_time))

    else:
        start_time = time.time()

        ls_metadata, dct_no_entries = fetch_by_imdb_ids(ls_imdb_ids)
        print_exception_statistic(dct_no_entries)

        print("--- %s seconds ---" % (time.time() - start_time))

    df_meta = pd.DataFrame(ls_metadata)
    print('Shape of crawled dataset:{}'.format(df_meta.shape[0]))
    return df_meta


        # tmp_list = []
        # for element in df_meta[column]:
        #
        #     ls_ls_casted = [eval(str_elem) for str_elem in df_meta[element].values]
        #     itertools.chain(*ls_ls_casted)
        #     if(type(element)==str):
        #         tmp_list.extend(eval(element))
        #    # tmp_list.value

        # dct_rel_freq[element] =
        # df_meta[column] = tmp_list

    # df = df_meta['cast'][0].value_counts()
    # print(df)

def imdb_id_2_full_Id(imdb_id):
    # str_imdb_id = row['imdbId'].astype(str)
    # if(len(str_imdb_id) >6):
    if (imdb_id >= 1000000):
        prefix = 'tt'
    elif(imdb_id >= 100000):
        prefix = 'tt0'
    elif(imdb_id >= 10000):
        prefix = 'tt00'
    else:
        prefix = 'tt000'

    return prefix + str(imdb_id)

def join_kaggle_with_links():
    df_movies_large = pd.read_csv('../data/kaggle/df_imdb_kaggle.csv')
    df_links = utils.load_dataset(small_dataset=True)

    # df_links['imdb_title_id'] = 'tt0'+df_links['imdbId'].astype(str) if
    for idx in range(df_links.shape[0]):  # iterrows does not preserve dtypes
        full_id = imdb_id_2_full_Id(df_links.loc[idx, 'imdbId'])
        df_links.loc[idx, 'imdb_title_id'] = full_id

    df_links_joined_one = df_links.set_index('imdb_title_id').join(df_movies_large.set_index('imdb_title_id'),
                                                                   on='imdb_title_id', how='left')
    df_links_joined_one.to_csv('../data/generated/df_joined_partly.csv', index_label='imdb_title_id')

def main():

    df_links_joined_one = pd.read_csv('../data/generated/df_joined_partly.csv')
    # df_links_joined_one.to_csv('../data/generated/df_links_kaggle.csv')
    # df_links_joined = df_links.merge(df_movies_large, on='imdb_title_id')

    # benchmark_string_comparison()
    print('<----------- Metadata Crawler has started ----------->')

    # fetch_example()
    # Load Dataset
    small_dataset = True
    multi_processing = True
    develop_size = 8
    metadata = None
    crawl = True
    from_scratch = True
    no_processes = 4
    df_links = utils.load_dataset(small_dataset=small_dataset)
    if (crawl):
        #Todo: Differentiate between the approach of extending the kaggle dataset and crawling the metadata for all ids from scratch

        if(from_scratch):
            ls_imdb_ids = list(df_links['imdbId'])
            ls_imdb_ids = [(id, True) for id in ls_imdb_ids]
        else:
            # Enhance existing dataset by fetching metadata
            ls_imdb_ids = list(df_links_joined_one.loc[~df_links_joined_one['title'].isna()]['imdbId'])  # list(df_links['imdbId'])
            ls_tpl_imdb_ids = [(id, False) for id in ls_imdb_ids]

            ls_missing_imdb_ids = list(df_links_joined_one.loc[df_links_joined_one['title'].isna()]['imdbId'])
            ls_tpl_missing_imdb_ids = [(id, True) for id in ls_missing_imdb_ids]
            ls_tpl_imdb_ids.extend(ls_tpl_missing_imdb_ids)
            ls_tpl_imdb_ids= ls_tpl_imdb_ids[6000:]
            # ls_imdb_ids = list(df_links_joined_one.index)

            # ls_crawl_from_scratch = [False] * len(ls_missing_imdb_ids)
            ls_imdb_ids = ls_tpl_imdb_ids

        df_meta = crawl_metadata(ls_imdb_ids,
                                 multi_processing=multi_processing,
                                 no_processes=no_processes,
                                 develop_size=develop_size
                                 )
        print('Fetching Metadata done.')
        df_meta = clean_movies(df_meta)

    else:
        df_meta = pd.read_csv('../data/input/movielens/small/df_movies.csv')
        utils.compute_relative_frequency(df_meta)
        df_meta.to_csv('../data/generated/df_movies_cleaned.csv')

    print('col df_links_joined_one:', len(df_links_joined_one))

    if(not from_scratch):
        for col in df_meta.columns:
            if(col not in df_links_joined_one.columns):
                df_links_joined_one[col]=""
        #Extend originial dataframe by columns of new one:
        for row_idx in range(df_meta.shape[0]):
            row = df_meta.loc[row_idx,]
            row['imdbId'] = int(row['imdbid'])
            imdb_id = row['imdbId']

            #set row
            df_links_joined_one.loc[df_links_joined_one['imdbId'] == imdb_id, df_meta.columns] = row

        # for col in df_meta.columns:
        #     df_links_joined_one[col]=df_meta[col]

        print('col df_links_joined_one:', len(df_links_joined_one))
        num_nans_before = len(ls_missing_imdb_ids)
        # df_links_joined_one.set_index('imdbId').update(df_meta.set_index('imdbId'))
        # df_meta = df_meta.fillna('missing')
        # df_links_joined_one = df_links_joined_one.update(df_meta, raise_conflict=True)

        # TODO Iterate through indizes of df_meta and update df_links_joined_one --> Should be obsolete once I pass all Ids to crawling

        print("Nans before:{}, Nans after joining:{} (Before must be greater)".format(num_nans_before, df_links_joined_one[
            'title'].isna().sum()))
        assert num_nans_before > df_links_joined_one['title'].isna().sum()

    # transform names to ids
    df_meta = utils.names2ids(df=df_meta, column_name=['cast', 'stars'])

    dct_attribute_distribution = utils.compute_relative_frequency(df_meta)
    # Save data
    print('Save enhanced movielens dataset...')
    if (small_dataset):
        # if(df_links_joined_one != None):
        #     df_links_joined_one
        df_meta.to_csv("../data/generated/df_movies_cleaned.csv")
        utils.save_dict_as_json(dct_attribute_distribution, 'attribute_distribution.json')
    else:
        df_meta.to_csv("../data/generated/df_movies_cleaned.csv")
    print('<----------- Processing finished ----------->')

def clean_movies(df_movies: pd.DataFrame):
    # clean data
    # df_movies = pd.read_csv('../data/generated/df_movies.csv')

    print('Removing data with more than 80% holding nans..')
    print('Shape before cleaning:{}'.format(df_movies.shape))
    df_cleaned = df_movies.dropna(axis=1, how='all')
    print('Sum of isNull for all columns: ', df_cleaned.isnull().sum())
    df_cleaned_two = df_cleaned.loc[:, df_cleaned.isnull().sum() < 0.8 * df_cleaned.shape[
        0]]  # ist asks: Which columns have less nans than 80% of the data? And those you have to keep
    print('Shape after cleaning is done: {}'.format(df_cleaned_two.shape))

    print('Columns before renaming: {}', df_cleaned_two.columns)
    df = janitor.clean_names(df_cleaned_two)
    print('Columns after renaming: {}', df.columns)
    df=df.fillna('missing')
    import re
    df['original_air_date'] = df['original_air_date'].apply(lambda x: re.findall('\d{4}',x))

    return df

    # df.to_csv('../data/generated/df_movies_cleaned.csv')
if __name__ == '__main__':
    main()
    # df_movies = pd.read_csv('../data/generated/df_movies_cleaned.csv')
    # df = clean_movies(df_movies)
    # df =df.remove_columns(['unnamed_0','unnamed_0'])
    # df.to_csv('../data/generated/df_movies_cleaned3.csv', index=False)


    dct_attribute_distribution = utils.compute_relative_frequency(pd.read_csv('../data/generated/df_movies_cleaned3.csv'))
    utils.save_dict_as_json(dct_attribute_distribution, 'attribute_distribution.json')
