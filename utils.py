import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pandas as pd
import numpy as np
from kmeans import KMeans

def evaluate_clustering_number(config, X, Y, dim_reduc = False):
    """
    evaluate_clustering_number
    Function that depending on the Clustering algorithm
    that has been selected computes different clustering
    metrics for different number of clusters. Results are
    saved in a csv file
    :param config: config dictionary
    :param X: nxd dataset
    :param Y: target
    :return eval dictionary which keys are the number of
    clusters and values another dictionary with the metrics
    for each number of cluster
    """
    eval = {}
    for n in range(2, config['max_num_clusters']):
        print('[INFO] Testing for k = {} clusters'.format(n))
        if config['clusteringAlg'] == 'agg':
            clustering = AgglomerativeClustering(n_clusters = n, affinity=config['affinity'], linkage=config['linkage'])
        if config['clusteringAlg'] == 'km':
            clustering = KMeans(n_clusters=n)

        labels = clustering.fit_predict(X)
        ari = adjusted_rand_score(Y, labels)
        sil = silhouette_score(X, labels)
        dbs = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        eval[n] = {'ari':round(ari, 3), 'sil': round(sil, 3), 'dbs': round(dbs, 3), 'ch': round(ch, 3)}

    #make_plots(config, eval)
    save_log(config, eval, dim_reduc)
    return eval

def save_log(config, eval, dim_reduc):
    """
    save_log
    Function that takes the eval dictionary and saves the results
    in an existing csv file that gets updated with the new results
    :param config: config dictionary
    :param eval: eval dictionary
    """
    path = './results/{}_{}.csv'.format(config['dataset'], config['dimReduction']) if dim_reduc else './results/{}.csv'.format(config['dataset'])
    data = [[k, v['ari'], v['sil'], v['dbs'], v['ch'] ] for k, v in eval.items()]
    if os.path.isfile(path):
        df = pd.read_csv(path)
        df_aux = pd.DataFrame(data, columns = ['Number of clusters', 'ari', 'sil', 'dbs', 'ch'])
        df_aux['dataset'] = config['dataset']
        df_aux['clusteringAlg'] = config['clusteringAlg']
        df_aux['affinity'] = config['affinity'] if config['clusteringAlg'] == 'agg' else 'None'
        df_aux['linkage'] = config['linkage'] if config['clusteringAlg'] == 'agg' else 'None'
        df_both = pd.concat([df, df_aux], ignore_index=True, sort=False)
        df_both = df_both.drop_duplicates()
        df_both.to_csv(path, index=False)
    else:
        df = pd.DataFrame(data, columns = ['Number of clusters', 'ari', 'sil', 'dbs', 'ch'])
        df['dataset'] = config['dataset']
        df['clusteringAlg'] = config['clusteringAlg']
        df['affinity'] = config['affinity'] if config['clusteringAlg'] == 'agg' else 'None'
        df['linkage'] = config['linkage'] if config['clusteringAlg'] == 'agg' else 'None'
        df.to_csv(path, index=False)

def make_plots(config, metric = 'sil'):
    metrics_info = {'sil': {'lim':(-1, 1), 'name':'Silhouette Score'},
                    'ari': {'lim':(-0.1, 1), 'name':'ARI'}}
    clustering_no_dimred = pd.read_csv('./results/{}.csv'.format(config['dataset']))
    clustering_pca = pd.read_csv('./results/{}_pca.csv'.format(config['dataset']))
    clustering_fa = pd.read_csv('./results/{}_fa.csv'.format(config['dataset']))
    alg = ['km', 'agg']
    agg_linkage = ['single', 'complete', 'average', 'ward']
    fig = plt.figure(figsize=(20,7))
    plt.subplot(1, 3, 1)
    plt.title('No dimensionality reduction')
    subset_km = clustering_no_dimred.loc[clustering_no_dimred['clusteringAlg'] == 'km']
    plt.plot(subset_km['Number of clusters'], subset_km[metric], linestyle='solid', marker='o', label='K-means')
    for clust_linkage in agg_linkage:
        subset = clustering_no_dimred.loc[(clustering_no_dimred['clusteringAlg'] == 'agg') & (clustering_no_dimred['linkage'] == clust_linkage)]
        plt.plot(subset['Number of clusters'], subset[metric], linestyle = 'solid', marker = 'o', label = 'Agg-'+clust_linkage)
    plt.grid(True)
    plt.ylim(metrics_info[metric]['lim'])
    plt.xlabel('Clusters')
    plt.ylabel(metrics_info[metric]['name'])
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.title('PCA')
    subset_km = clustering_pca.loc[clustering_pca['clusteringAlg'] == 'km']
    plt.plot(subset_km['Number of clusters'], subset_km[metric], linestyle='solid', marker='o', label='K-means')
    for clust_linkage in agg_linkage:
        subset = clustering_pca.loc[(clustering_pca['clusteringAlg'] == 'agg') & (clustering_pca['linkage'] == clust_linkage)]
        plt.plot(subset['Number of clusters'], subset[metric], linestyle = 'solid', marker = 'o', label = 'Agg-'+clust_linkage)
    plt.grid(True)
    plt.ylim(metrics_info[metric]['lim'])
    plt.xlabel('Clusters')
    plt.ylabel(metrics_info[metric]['name'])
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.title('Factor Analysis')
    subset_km = clustering_tsne.loc[clustering_tsne['clusteringAlg'] == 'km']
    plt.plot(subset_km['Number of clusters'], subset_km[metric], linestyle='solid', marker='o', label='K-means')
    for clust_linkage in agg_linkage:
        subset = clustering_tsne.loc[(clustering_tsne['clusteringAlg'] == 'agg') & (clustering_tsne['linkage'] == clust_linkage)]
        plt.plot(subset['Number of clusters'], subset[metric], linestyle = 'solid', marker = 'o', label = 'Agg-'+clust_linkage)
    plt.grid(True)
    plt.ylim(metrics_info[metric]['lim'])
    plt.xlabel('Clusters')
    plt.ylabel(metrics_info[metric]['name'])
    plt.legend()
    plt.savefig('./plots/clustering_comparison_{}_{}.jpg'.format(config['dataset'], metric), dpi=350, bbox_inches='tight')
    plt.close()