
import json
import seaborn as sns
import pandas as pd
from sklearn import decomposition
import math

from datetime import datetime
import random
from tqdm import tqdm
import ast
import itertools
from collections import Counter
import plotly.express as px
import matplotlib.pyplot as plt
import os
import numpy as np

blue = '#5275A8'
orange = '#D7865B'
green = '#47A169'


def create_heatmap(np_arr, title, y_label, x_label,file_name, experiment_path, dct_params):
    ax = sns.heatmap(np_arr)
    # plt.suptitle(title)
    ax.set(title=title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    save_figure(ax.get_figure(), experiment_path, file_name, dct_params)
    plt.show()



def save_figure(fig, experiment_path, name, dct_params):
    if(dct_params==None):
        fig.savefig(experiment_path + name + ".png")
    else:
        image_name = name + "_" + "_".join(str(val)[:4]+"_"+key for key, val in dct_params.items())
        fig.savefig(experiment_path + image_name + ".png")


def plot_mce(model, neptune_logger, max_epochs):
    avg_mce = model.avg_mce

    ls_x = []
    ls_y = []

    for key, val in avg_mce.items():
        neptune_logger.log_metric('MCE_' + key, val)
        ls_x.append(key)
        ls_y.append(val)
    plt.figure()
    fig, ax = plt.subplots(figsize=(20, 12))

    sns_plot = sns.barplot(x=ls_x, y=ls_y)
    fig = sns_plot.get_figure()
    # fig.set_xticklabels(rotation=45)
    plt.xticks(rotation=70)
    plt.tight_layout()
    fig.savefig("./results/images/mce_epochs_" + str(max_epochs) + ".png")

def ls_columns_to_dfrows(ls_val, column_base_name, model):
    print(ls_val.shape)

    ls_columns = [column_base_name + str(i) for i in range(1, ls_val.shape[1] + 1)]
    # print(ls_columns)
    df_z = pd.DataFrame(data=ls_val, columns=ls_columns)
    df_z['y'] = model.test_y
    # print(df_z.columns)
    df_piv = df_z.melt(id_vars=['y'],var_name='cols', value_name='values') # Transforms it to: _| cols | vals|
    return df_piv

def plot_catplot(df, title, experiment_path, dct_params):
    # plt.xticks(rotation=45)
    g=sns.catplot(x="cols", y="values",s=3, data=df).set(title=title)
    g.ax.set_xticklabels(g.ax.get_xticklabels(), rotation=65)
    save_figure(g, experiment_path, 'catplot', dct_params)

    plt.show()

def plot_swarmplot(df, title, experiment_path,dct_params):
    # plt.xticks(rotation=45)
    ax=sns.swarmplot(x="cols", y="values",s=3, data=df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=65)
    ax.set(title=title)
    plt.show()
    save_figure(ax.get_figure(), experiment_path, 'swarmplot', dct_params)


def plot_violinplot(df, title,experiment_path,dct_params):
    # plt.xticks(rotation=45)
    ax=sns.violinplot(x="cols", y="values", data=df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=65)
    ax.set(title=title)

    save_figure(ax.get_figure(), experiment_path, 'violinplot', dct_params)
    plt.show()

def plot_distribution(df_melted, title, experiment_path, dct_params):
    # plt.figure(figsize=(10,10))
    # sns.violinplot(x=foo[:,0])
    fig = sns.displot(df_melted, x="values", hue="cols", kind="kde", rug=True).fig
    fig.suptitle(title)
    plt.xlabel('z values')
    save_figure(fig, experiment_path, 'distribution', dct_params)

    plt.show()

#PCA
def apply_pca(np_x):
    if(np_x.shape[1] > 1):
        pca = decomposition.PCA(n_components=2)
        # print(np_x)
        pca.fit(np_x)
        X = pca.transform(np_x)
        # print(X)
        return X

def plot_2d_pca(np_x, title, experiment_path, dct_params):
    df_pca = pd.DataFrame(np_x, columns=['pca_1', 'pca_2'])
    fig = sns.scatterplot(data=df_pca, x="pca_1", y="pca_2").get_figure()
    fig.suptitle(title)

    # g.set_xticklabels(g.get_xticklabels(), rotation=45)
    # plt.setp(ax.get_xticklabels(), rotation=45)
    # plt.scatter(np_x[:,0], np_x[:,1]) #only on numpy array
    save_figure(fig, experiment_path,'pca', dct_params)
    plt.show()

def plot_mce_wo_kld(df_mce, title, experiment_path, dct_params):
    plt.figure(figsize=(30, 20))
    df_mce['category'] = df_mce.index
    df_piv = df_mce.melt(id_vars='category', var_name='latent_factor', value_name='mce')
    fig = sns.catplot(x='latent_factor',
                      y='mce',
                      hue='category',
                      data=df_piv,
                      kind='bar',
                      legend_out=False,
                      aspect=1.65
                      )
    plt.title(title, fontsize=17, y=1.08)

    plt.legend(bbox_to_anchor=(0.5, -0.25),  # 1.05, 1
               loc='upper center',  # 'center left'
               borderaxespad=0.,
               fontsize=8,
               ncol=5)

    plt.tight_layout()
    save_figure(fig, experiment_path,title, dct_params)
    plt.show()

def plot_mce_by_latent_factor(df_mce, title, experiment_path, dct_params):


    plt.figure(figsize=(30, 20))
    df_mce['category'] = df_mce.index
    df_piv = df_mce.melt(id_vars='category', var_name='latent_factor', value_name='mce')
    fig = sns.catplot(x='latent_factor',
                      y='mce',
                      hue='category',
                      data=df_piv,
                      kind='bar',
                      legend_out=False,
                      aspect=1.65,
                      palette=[orange, blue, green]
                      )
    plt.title(title, fontsize=17, y=1.08)

    plt.legend(bbox_to_anchor=(0.5, -0.25),  # 1.05, 1
               loc='upper center',  # 'center left'
               borderaxespad=0.,
               fontsize=8,
               ncol=5)

    plt.tight_layout()
    save_figure(fig, experiment_path,'mce-latent-factor', dct_params)
    plt.show()

def plot_ig_by_latent_factor(df_ig, title, experiment_path, dct_params):
    sns.set_theme(style="darkgrid")

    # plt.figure(figsize=(30, 20))
    with sns.plotting_context("notebook",font_scale=1.5):
        fig, ax = plt.subplots(figsize=(30, 20))
        df_piv = df_ig.melt(id_vars=['spectrum', 'LF'], var_name='attribute', value_name='Information Gain')
        g = sns.catplot(x="spectrum", y="Information Gain",
                        hue="attribute", col="LF",
                        data=df_piv, kind="bar",
                        ax=ax,
                        # col_wrap=5,
                        #   height=4,
                        # height=4, aspect=.7);
                        # legend_out=True,
                          legend=False,
                        aspect=.6,
                        palette=[green, blue, orange]
                        );

        for ax in g.axes.flat:
            box = ax.get_position()
            ax.set_position([box.x0+0.01, box.y0+0.15, box.width , box.height* 0.85])

        # plt.legend(loc='lower left',  bbox_to_anchor=(0,1.02,1,0.2),borderaxespad=0, mode="expand", ncol=3)#bbox_transform=fig.transFigure,
        #https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
        #https://stackoverflow.com/questions/38773560/how-to-move-the-legend-in-seaborn-facetgrid-outside-of-the-plot
        plt.legend(bbox_to_anchor=(0.175, 0), loc="lower left",
                   bbox_transform=fig.transFigure, ncol=3)
        # (fig.set(ylim=(0, 1))
        #  .despine(left=True))
        # plt.subplots_adjust(hspace=0.4, wspace=0.4)
        # plt.suptitle('f', fontsize=17, y=1.08)
        # g.fig.suptitle('Title',fontsize=17, y=1)
        # plt.legend(bbox_to_anchor=(1, 0),  # 1.05, 1
        #            loc='lower right',  # 'center left'
        #            mode="expand",
        #            borderaxespad=0.,
        #            fontsize=14,
        #            ncol=5)
        # g.fig.legend(loc=7)
        #
        # g.fig.tight_layout()
        # g.fig.subplots_adjust(right=0.75)
        # plt.legend()
        plt.tight_layout()
        save_figure(g, experiment_path,'ig-latent-factor', dct_params)
        plt.show()
    print('f')

def plot_parallel_plot(df_mce, title, experiment_path, dct_params):
    df_flipped = df_mce.transpose() # columns and rows need to be flipped
    ls_lf = [i for i in range(0, df_flipped.shape[0])]
    df_flipped['latent_factors'] = ls_lf
    ax = pd.plotting.parallel_coordinates(
        df_flipped, 'latent_factors', colormap='viridis')

    ax.xaxis.set_tick_params(labelsize=6)
    fig = ax.get_figure()
    plt.title(title, fontsize=20, y=1.08)
    # plt.xlabel('xlabel', fontsize=10)
    plt.xticks(rotation=90)

    plt.tight_layout()
    save_figure(fig, experiment_path,'parallel-plot', dct_params)
    plt.show()

def plot_KLD(ls_kld, title, experiment_path, dct_params):
    # ls_kld =[2,200,2000,1800,2000,1500]
    df_kld = pd.DataFrame(data=ls_kld, columns=['KLD'])
    ax = sns.lineplot(data = df_kld, x=df_kld.index, y="KLD")


    plt.show()


def plot_pairplot_lf_kld(model, title, experiment_path, dct_params):
    df_kld_matrix = pd.DataFrame(data=model.kld_matrix_train,
                                 columns=[str(i) for i in range(0, model.kld_matrix_train.shape[1])])
    fig = sns.pairplot(df_kld_matrix, corner=True, aspect=1.65).fig
    # for ax in fig.axes:
    #     ax.set_xlim(-3, 3)
    #     ax.set_ylim(-3, 3)


    fig.suptitle(title)
    # plt.title(title, fontsize=17, y=1.08)
    plt.tight_layout()
    plt.ylabel('Latent Factor')
    plt.xlabel('Latent Factor')
    save_figure(fig, experiment_path, 'lf-correlation-kld', dct_params)
    plt.show()

def plot_kld_line_of_latent_factor(model, title, experiment_path, dct_params):
    df_kld_matrix_test = pd.DataFrame(data=model.kld_matrix_test,
                                     columns=[str(i) for i in range(0, model.kld_matrix_test.shape[1])])
    ax = df_kld_matrix_test.plot.line()

    save_figure(ax.get_figure(), experiment_path, 'kld-lf-test', dct_params)

    plt.show()

def plot_kld_mean_of_latent_factor(model, title, experiment_path, dct_params):
    df_kld_matrix_train = pd.DataFrame(data=model.kld_matrix_train,
                                 columns=[str(i) for i in range(0, model.kld_matrix_train.shape[1])])

    df = df_kld_matrix_train.agg("mean")  # .reset_index()


    ax = df.plot.bar(stacked=False, rot=0)
    plt.title(title +' // Train', fontsize=17, y=1.08)
    plt.ylabel('KLD')
    plt.xlabel('Latent Factor')
    fig = ax.get_figure()
    save_figure(fig, experiment_path, 'kld-lf-mean-train', dct_params)

    plt.show()

    df_kld_matrix_test = pd.DataFrame(data=model.kld_matrix_test,
                                      columns=[str(i) for i in range(0, model.kld_matrix_test.shape[1])])
    df = df_kld_matrix_test.agg("mean")  # .reset_index()

    ax = df.plot.bar(stacked=False, rot=0)
    plt.title(title + ' // Test', fontsize=17, y=1.08)
    plt.ylabel('KLD')
    plt.xlabel('Latent Factor')
    fig = ax.get_figure()
    save_figure(fig, experiment_path, 'kld-lf-mean-test', dct_params)

    plt.show()



def plot_pairplot_lf_z(model, title, experiment_path, dct_params):

    # sns.set_style("darkgrid")
    df_z_matrix = pd.DataFrame(data=model.np_z_test,
                                 columns=[str(i) for i in range(0, model.np_z_test.shape[1])])
    if(model.used_data =='dsprites'):
        lf_cols = [str(i) for i in range(model.np_z_test.shape[1])]
        ls_gen_facs_partially = list(model.dsprites_lat_names)
        ls_gen_facs = ['y_' + name for name in ls_gen_facs_partially]
        ls_gen_facs.insert(0,'y_white')
        for idx, name in enumerate(ls_gen_facs):
            df_z_matrix[name] = model.test_y[:, idx]

        df_z_matrix = df_z_matrix.drop(['y_scale', 'y_white', 'y_scale', 'y_orientation', 'y_posX', 'y_posY'], axis=1)
        # df_z_matrix = df_z_matrix.melt(id_vars=ls_gen_facs,value_vars=lf_cols, var_name='lf',
        #                              value_name='values')  # Transforms it to: _| cols | vals|

    # elif(model.used_data =='morpho'):
    #     df_z_matrix['y'] = model.test_y
    else:
        if(model.np_synthetic_data is not None):
            df_z_matrix['y'] = model.test_y

        # raise NotImplementedError
    with sns.plotting_context("notebook", font_scale=2.5):
        if(model.used_data == 'dsprites'):
            df = df_z_matrix.sample(frac=1, random_state=42).reset_index(drop=True)[:1000]
            fig = sns.pairplot(df, corner=True, aspect=1.65, hue='y_shape').fig
        else:
            if (model.np_synthetic_data is None):
                g = sns.pairplot(df_z_matrix, corner=True, aspect=1.65, diag_kind="kde")
                fig = g.fig
                g.map_lower(sns.kdeplot, levels=4, color=".2")
            else:
                fig = sns.pairplot(df_z_matrix, corner=True, aspect=1.65, hue='y').fig

        fig.suptitle(title)
        # plt.title(title, fontsize=17, y=1.08)
        plt.tight_layout()
        # plt.setp(fig._legend.get_title(), fontsize=20)
        save_figure(fig, experiment_path, 'lf-correlation-z', dct_params)
        plt.show()


def plot_3D_lf_z(model, title, experiment_path, dct_params):
    # df = px.data.iris()
    df_z_matrix = pd.DataFrame(data=model.np_z_test,
                               columns=[str(i) for i in range(0, model.np_z_test.shape[1])])
    df_z_matrix['y'] = model.test_y
    if(df_z_matrix.shape[1]==3):
        fig = px.scatter_3d(df_z_matrix, x='0', y='1', color='y', opacity=0.7)
    else:
        fig = px.scatter_3d(df_z_matrix, x='0', y='1', z='2', color='y', opacity=0.7)

    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    # plt.tight_layout()
    fig.show()


def polar_plot(model, title, experiment_path, dct_params):

    df_z_matrix = pd.DataFrame(data=model.np_z_test,
                               columns=[str(i) for i in range(0, model.np_z_test.shape[1])])
    # if(df_z_matrix.shape[1]!=2):
    #     df_z_matrix = df_z_matrix.iloc[:,1:3]
        # return

    df_z_matrix['y'] = model.test_y

    # xs = np.arange(7)
    # ys = xs ** 2

    fig = plt.figure(figsize=(5, 10))
    plt.title('Cartesian to Polar Coordinates for the first two Latent Factors')
    ax = plt.subplot(2, 1, 1)

    # If we want the same offset for each text instance,
    # we only need to make one transform.  To get the
    # transform argument to offset_copy, we need to make the axes
    # first; the subplot command above is one way to do this.
    # trans_offset = mtransforms.offset_copy(ax.transData, fig=fig,
    #                                        x=0.05, y=0.10, units='inches')

    ls_zero = df_z_matrix['0']
    ls_one = df_z_matrix['1']


    for x, y, factor in zip(ls_zero, ls_one, df_z_matrix['y']):
        if(factor =='genres'):
            plt.plot(x, y, 'ro')

        else:
            plt.plot(x, y, 'bo')

        # plt.text(x, y, '%d, %d' % (int(x), int(y)))

    # offset_copy works for polar plots also.
    ax = plt.subplot(2, 1, 2, projection= None) #projection= polar

    # trans_offset = mtransforms.offset_copy(ax.transData, fig=fig,
    #                                        y=6, units='dots')

    for xs, ys, factor in zip(ls_zero, ls_one, df_z_matrix['y']):
        r = np.sqrt(xs ** 2 + ys ** 2)
        t = np.arctan2(ys, xs)

        # plt.polar(t, r, 'ro')
        if (factor == 'genres'):
            plt.plot(t, r, 'ro')

        else:
            plt.plot(t, r, 'bo')
        # plt.text(xs, ys, '%d, %d' % (int(x), int(y)), horizontalalignment='center', verticalalignment='bottom')

    save_figure(fig, experiment_path, 'polar', dct_params)
    plt.show()

def plot_z_values_of_latent_factors(model, title, experiment_path, dct_params, sum = False):
    df_z_matrix = pd.DataFrame(data=model.np_z_test,
                                 columns=[str(i) for i in range(0, model.np_z_test.shape[1])])

    if(model.used_data =='dsprites'):

        ls_gen_facs_partially = list(model.dsprites_lat_names)
        ls_gen_facs = ['y_'+ name for name in ls_gen_facs_partially]
        ls_gen_facs.insert('y_whitey',0)
        for idx, name in enumerate(ls_gen_facs):
            df_z_matrix[name] = model.test_y[:,idx]

        print('fo')
        # import plotly.graph_objects as go
        #
        #
        # fig = go.Figure()
        # fig.add_trace(go.Bar(
        #     x=ls_gen_facs,
        #     y=[20, 14, 25, 16, 18, 22, 19, 15, 12, 16, 14, 17],
        #     name='Primary Product',
        #     marker_color='indianred'
        # ))
        # fig.add_trace(go.Bar(
        #     x=months,
        #     y=[19, 14, 22, 14, 16, 19, 15, 14, 10, 12, 12, 16],
        #     name='Secondary Product',
        #     marker_color='lightsalmon'
        # ))

        # Here we modify the tickangle of the xaxis, resulting in rotated labels.
        # fig.update_layout(barmode='group', xaxis_tickangle=-45)
        # plt.show()
    else:
        df_z_matrix['y'] = model.test_y
        df = df_z_matrix.groupby('y').agg("mean")#.reset_index()



    #Create plot for generative factors on the x-axis
    ax = df.plot.bar(stacked=False,rot=0)

    plt.title('Mean z-values for Generative Factors c', fontsize=17, y=1.08)
    plt.tight_layout()
    fig = ax.get_figure()
    plt.ylabel('Mean Value')
    plt.xlabel('Generative Factor c')
    save_figure(fig, experiment_path, 'gen-factors2latent-factors-mean', dct_params)
    plt.show()

    # Create plot for latent factors on the x-axis
    df['c'] = df.index
    df = df.melt(id_vars='c')
    ax = sns.barplot(data=df, x='variable', y='value', hue='c')

    # df=df.T
    # ax = df.plot.bar(stacked=False, rot=0)
    plt.title('Mean z-value for Generative Factors c', fontsize=17, y=1.08)
    plt.tight_layout()
    fig = ax.get_figure()
    plt.ylabel('Mean Value')
    plt.ylabel('Latent Factor (z)')
    save_figure(fig, experiment_path, 'latent-factors2gen-factors-mean', dct_params)
    plt.show()

    #Do the same vor sum
    if(sum):
        df_two = df_z_matrix.groupby('y').agg("sum")#.reset_index()
        ax = df_two.plot.bar(stacked=False,rot=0)
        fig = ax.get_figure()
        plt.title('gen-factors2latent-factors-sum', fontsize=17, y=1.08)
        plt.ylabel('z value')
        save_figure(fig, experiment_path, 'gen-factors2latent-factors-sum', dct_params)
        plt.show()

def plot_parallel_lf(model,experiment_path, dct_params):

        df_z_matrix = pd.DataFrame(data=model.np_z_test,
                                   columns=[str(i) for i in range(0, model.np_z_test.shape[1])])
        df_z_matrix['y'] = model.test_y
        ax = pd.plotting.parallel_coordinates(
            df_z_matrix, 'y', colormap='viridis')
        plt.xlabel("Latent Factor")
        plt.ylabel("z value")
        plt.title('Parallel Plot for Latent Factors and Generative Factor ')
        save_figure(ax.get_figure(), experiment_path, 'parrallel-latent-generative', dct_params)

        plt.show()

def swarm_plot_melted(df_melted,experiment_path_test, dct_params):
    fig = sns.catplot(x="cols", y="values", hue="y", kind="swarm", data=df_melted)
    plt.title('Swarm Plot for Generative Factors')
    save_figure(fig, experiment_path_test, 'swarm-plot-z-generative', dct_params)

    plt.show()



def plot_movies_combined_with_z(model, experiment_path, dct_params):
    lf_cols = [i for i in range(model.np_z_test.shape[1])]
    df = model.df_movies_z_combined
    df_all_one = df.groupby(['year', 'rating', 'genres']).agg('mean')
    df_all_two = df.groupby([ 'rating', 'genres', 'year']).agg('mean')
    df_all_three = df.groupby([ 'genres','year', 'rating',]).agg('mean')
    df_all_one['x'] = df_all_one.index.values
    df_all_two['x'] = df_all_two.index.values
    df_all_three['x'] = df_all_three.index.values
    df_year = df.groupby('year').agg('mean').reset_index()
    df_rating = df.groupby('rating').agg('mean').reset_index()
    df_genre = df.groupby('genres').agg('mean').reset_index()


    df_all_one= df_all_one.melt(id_vars='x',value_vars=lf_cols, var_name='lf', value_name='values')   # Transforms it to: _| cols | vals|
    df_all_two= df_all_two.melt(id_vars='x',value_vars=lf_cols, var_name='lf', value_name='values')   # Transforms it to: _| cols | vals|
    df_all_three= df_all_three.melt(id_vars='x',value_vars=lf_cols, var_name='lf', value_name='values')   # Transforms it to: _| cols | vals|
    df_year= df_year.melt(id_vars='year',value_vars=lf_cols, var_name='lf', value_name='mean value')   # Transforms it to: _| cols | vals|
    df_rating = df_rating.melt(id_vars='rating',value_vars=lf_cols, var_name='lf', value_name='mean value')   # Transforms it to: _| cols | vals|
    df_genre = df_genre.melt(id_vars='genres',value_vars=lf_cols, var_name='lf', value_name='mean value')   # Transforms it to: _| cols | vals|

    sns.set(style="darkgrid", rc={"lines.linewidth": 0.9}) #style ="ticks"
    def plot_groups(df, title):
        fig =sns.catplot(x="x", y="values", hue="lf", kind="point", data=df, aspect=3, legend_out=True, plot_kws=dict( linestyles=["-", "--"]))
        plt.xticks(rotation=80)
        plt.show()
        save_figure(fig, experiment_path, title, dct_params)

    plot_groups(df_all_one, 'groupby-movie-all-z-one')
    plot_groups(df_all_two, 'groupby-movie-all-z-two')
    plot_groups(df_all_three, 'groupby-movie-all-z-three')

    fig = sns.catplot(x="year", y="mean value", hue="lf", kind="point", data=df_year, aspect=1.5, legend=False)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="Latent\nFactor\n   (z)")
    # g.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    save_figure(fig, experiment_path, 'groupby-movie-year-z', dct_params)

    fig = sns.catplot(x="rating", y="mean value", hue="lf", kind="point", data=df_rating, aspect=1.5, legend=False)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="Latent\nFactor\n   (z)")
    plt.show()
    save_figure(fig, experiment_path, 'groupby-movie-rating-z', dct_params)

    fig = sns.catplot(x="genres", y="mean value", hue="lf", kind="point", data=df_genre, aspect=1.5, legend=False)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="Latent\nFactor\n   (z)")
    plt.show()
    save_figure(fig, experiment_path, 'groupby-movie-genre-z', dct_params)

def plot_covariance_heatmap(model, exp_img_path, dct_params):
    # logvar_mean_vec_train  = model.np_logvar_train.mean(axis=0)
    logvar_mean_vec_test = model.np_logvar_test.mean(axis=0)

    # create_heatmap(logvar_mean_vec_train, 'logvar train', 'dimension', 'dimension', exp_img_path, dct_params)
    create_heatmap(logvar_mean_vec_test, 'logvar test', 'dimension', 'dimension', exp_img_path, dct_params)



def plot_variance(model, experiment_path_test, experiment_path_train, dct_params):
    # np_var = np.exp(model.np_logvar_train)
    np_var = model.np_var_train
    df_sigma = pd.DataFrame(np_var)
    g = df_sigma.plot.line()
    # df_sigma['epochs'] = df_sigma.index
    # df_sigma = pd.melt(df_sigma, var_name="LF", value_name='variance', id_vars='epochs')
    # g = sns.lineplot(data=df_sigma, x="epochs", y ='variance', hue='LF')
    g.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Latent\n Factor')

    plt.ylabel('Variance')
    plt.xlabel('Epochs')
    plt.title('Variance of Latent Factors (Z) Train')
    plt.show()
    save_figure(g.figure,experiment_path_test,'variance_train', dct_params)

    df_z_matrix = pd.DataFrame(data=model.np_z_test,
                               columns=[str(i) for i in range(0, model.np_z_test.shape[1])])
    variance = df_z_matrix.var(axis=0)
    print("Variance: {} ".format(variance))
    np_var_test = model.np_var_test
    df_sigma = pd.DataFrame(np_var_test)
    g = df_sigma.plot.line()
    # df_sigma['epochs'] = df_sigma.index
    # df_sigma = pd.melt(df_sigma, var_name="LF", value_name='variance', id_vars='epochs')
    # g = sns.lineplot(data=df_sigma, x="epochs", y ='variance', hue='LF')
    g.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Latent\n Factor')

    plt.ylabel('Variance')
    plt.xlabel('Epochs')
    plt.title('Variance of Latent Factors (Z) Test')
    plt.show()
    save_figure(g.figure, experiment_path_test, 'variance_test', dct_params)

