import argparse
from datasets import preprocess_vote, preprocess_hypothyroid
from dim_reduction import PCA
from sklearn.manifold import TSNE
from visualize import plot_scores_2d, plot_scores_3d, plot_density, plot_loadings
from kmeans import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import FactorAnalysis
from utils import evaluate_clustering_number, make_plots
import numpy as np
parser = argparse.ArgumentParser()

### run--> python main.py --dataset vote
parser.add_argument("--dataset", type=str, default='vote', choices=['vote', 'hyp'])
parser.add_argument("--dimReduction", type=str, default='agg', choices=['pca', 'fa'])
parser.add_argument("--tsne", type=bool, default=True)
parser.add_argument("--num_dimensions", type=int, default=3)
parser.add_argument("--clusteringAlg", type=str, default='agg', choices=['km', 'agg'])
parser.add_argument("--max_num_clusters", type=int, default=7, choices=range(2,100))
parser.add_argument("--visualize_results", type=bool, default=False)
parser.add_argument("--plot_scores_colored_by_cluster", type=bool, default=False)
# For Agglomerative clustering parameters
parser.add_argument("--affinity", type=str, default = 'euclidean', choices=['euclidean', 'cosine'])
parser.add_argument("--linkage", type=str, default = 'ward', choices=['ward', 'complete', 'average', 'single'])
con = parser.parse_args()

def configuration():
    config = {
                'dataset':con.dataset,
                'dimReduction': con.dimReduction,
                'num_dimensions': con.num_dimensions,
                'clusteringAlg':con.clusteringAlg,
                'affinity': con.affinity,
                'linkage': con.linkage,
                'max_num_clusters': con.max_num_clusters,
                'visualize_results': con.visualize_results,
                'plot_scores_colored_by_cluster' : con.plot_scores_colored_by_cluster,
                'tsne':con.tsne
             }
    return config

def main():
    config = configuration()

    if config['visualize_results']:
        make_plots(config, metric = 'sil')
        make_plots(config, metric = 'ari')
        make_plots(config, metric = 'dbs')
        make_plots(config, metric = 'ch')
        return

    ### load dataset
    if config['dataset'] == 'vote':
        X, Y = preprocess_vote()

    elif config['dataset'] == 'hyp':
        X, Y = preprocess_hypothyroid()

    # perform clustering analysis without pca nor tsne
    if config['clusteringAlg'] == 'km':
        evaluate_clustering_number(config, X.values, Y)
        if config['dataset'] == 'vote':
            # run evaluation of number of clusters
            cluster_no_dimred = KMeans(2).fit_predict(X.values)
            np.save('./results/{}_{}.npy'.format(config['clusteringAlg'], config['dataset']), cluster_no_dimred)
        if config['dataset'] == 'hyp':
            cluster_no_dimred = KMeans(2).fit_predict(X.values)
            np.save('./results/{}_{}.npy'.format(config['clusteringAlg'], config['dataset']), cluster_no_dimred)

    if config['clusteringAlg'] == 'agg':
        evaluate_clustering_number(config, X, Y)
        if config['dataset'] == 'vote':
            cluster_no_dimred = AgglomerativeClustering(n_clusters = 2, affinity='euclidean', linkage='complete').fit_predict(X.values)
            np.save('./results/{}_{}.npy'.format(config['clusteringAlg'], config['dataset']), cluster_no_dimred)
        if config['dataset'] == 'hyp':
            cluster_no_dimred = AgglomerativeClustering(n_clusters = 2, affinity='euclidean', linkage='single').fit_predict(X.values)
            np.save('./results/{}_{}.npy'.format(config['clusteringAlg'], config['dataset']), cluster_no_dimred)

    best_configs = {
        'vote':{'pca':{'kmeans':[3], 'agg':[2, 'complete']},
                'fa': {'kmeans': [3], 'agg': [2, 'complete']},
                'tsne':{'kmeans':[2],'agg':[2, 'ward']}},
        'hyp': {'pca': {'kmeans': [3], 'agg': [2, 'complete']},
                'fa': {'kmeans': [3], 'agg': [2, 'complete']},
                'tsne': {'kmeans': [2], 'agg': [2, 'ward']}}
    }

    # perform dimensionality reduction
    if config['dimReduction'] == 'pca':
        pca = PCA(X.values, config['num_dimensions'], savefig = './plots/{}/pca/'.format(config['dataset']), verbose = True)
        scores, n_com_90_ex_var = pca.fit_transform()
        # perform clustering analysis
        evaluate_clustering_number(config, scores[:,0:n_com_90_ex_var], Y, dim_reduc=True)
        if config['clusteringAlg'] == 'km':
            cluster_dimred = KMeans(best_configs[config['dataset']]['pca']['kmeans'][0]).fit_predict(scores[:,0:n_com_90_ex_var])
            np.save('./results/pca_{}_{}.npy'.format(config['clusteringAlg'], config['dataset']), cluster_dimred)
        if config['clusteringAlg'] == 'agg':
            cluster_dimred = AgglomerativeClustering(n_clusters=best_configs[config['dataset']]['pca']['agg'][0],
                                                     affinity='euclidean',
                                                     linkage=best_configs[config['dataset']]['pca']['agg'][1]).fit_predict(scores[:,0:n_com_90_ex_var])
            np.save('./results/pca_{}_{}.npy'.format(config['clusteringAlg'], config['dataset']), cluster_dimred)

        loadings = pca.loadings
        plot_loadings(loadings, X.columns, savefig = './plots/{}/pca/'.format(config['dataset']), dim_1=1, dim_2=2)
        plot_loadings(loadings, X.columns, savefig='./plots/{}/pca/'.format(config['dataset']), dim_1=1, dim_2=3)
        plot_loadings(loadings, X.columns, savefig='./plots/{}/pca/'.format(config['dataset']), dim_1=2, dim_2=3)
        if config['dataset'] == 'vote':
            replace_vote = {0: 'republican', 1: 'democrat'}
            plot_scores_2d(scores, Y.replace(replace_vote).values, savefig = './plots/{}/pca/target_'.format(config['dataset']))
            plot_density(scores, Y.replace(replace_vote).values, dim = 1, savefig = './plots/{}/pca/target_'.format(config['dataset']))
            plot_density(scores, Y.replace(replace_vote).values, dim = 2, savefig='./plots/{}/pca/target_'.format(config['dataset']))
        elif config['dataset'] == 'hyp':
            replace_hyp = {0:'negative', 1:'compensated_hypothyroid', 2:'primary_hypothyroid', 3:'secondary_hypothyroid'}
            plot_scores_3d(scores, Y.replace(replace_hyp).values, savefig='./plots/{}/pca/target_'.format(config['dataset']))
            plot_scores_2d(scores, Y.replace(replace_hyp).values, savefig = './plots/{}/pca/target_'.format(config['dataset']))
            plot_scores_2d(scores, Y.replace(replace_hyp).values, savefig='./plots/{}/pca/target_'.format(config['dataset']), dim_1=1, dim_2=3)
            plot_scores_2d(scores, Y.replace(replace_hyp).values, savefig='./plots/{}/pca/target_'.format(config['dataset']), dim_1=2, dim_2=3)
            plot_density(scores, Y.replace(replace_hyp).values, dim = 1, savefig = './plots/{}/pca/target_'.format(config['dataset']))
            plot_density(scores, Y.replace(replace_hyp).values, dim = 2, savefig='./plots/{}/pca/target_'.format(config['dataset']))

    if config['dimReduction'] == 'fa':
        fa = FactorAnalysis(config['num_dimensions'], random_state=0)
        scores = fa.fit_transform(X.values)
        # perform clustering analysis
        evaluate_clustering_number(config, scores, Y, dim_reduc=True)
        if config['clusteringAlg'] == 'km':
            cluster_dimred = KMeans(best_configs[config['dataset']]['fa']['kmeans'][0]).fit_predict(scores)
            np.save('./results/fa_{}_{}.npy'.format(config['clusteringAlg'], config['dataset']), cluster_dimred)
        if config['clusteringAlg'] == 'agg':
            cluster_dimred = AgglomerativeClustering(n_clusters=best_configs[config['dataset']]['fa']['agg'][0],
                                                     affinity='euclidean',
                                                     linkage=best_configs[config['dataset']]['fa']['agg'][1]).fit_predict(scores)
            np.save('./results/fa_{}_{}.npy'.format(config['clusteringAlg'], config['dataset']), cluster_dimred)

    if config['tsne']:
        # compute T-SNE
        X_embedded = TSNE(n_components=config['num_dimensions'], learning_rate='auto', init = 'random', perplexity = 50,
                          random_state = 34).fit_transform(X.values)

        if config['dataset'] == 'vote':
            replace_vote = {0: 'republican', 1: 'democrat'}
            plot_scores_2d(X_embedded, Y.replace(replace_vote).values, savefig = './plots/{}/tsne/target_'.format(config['dataset']), tsne=True)
            plot_density(X_embedded, Y.replace(replace_vote).values, dim = 1, savefig = './plots/{}/tsne/target_'.format(config['dataset']), tsne=True)
            plot_density(X_embedded, Y.replace(replace_vote).values, dim = 2, savefig='./plots/{}/tsne/target_'.format(config['dataset']), tsne=True)
        elif config['dataset'] == 'hyp':
            replace_hyp = {0:'negative', 1:'compensated_hypothyroid', 2:'primary_hypothyroid', 3:'secondary_hypothyroid'}
            plot_scores_3d(X_embedded, Y.replace(replace_hyp).values, savefig='./plots/{}/tsne/target_'.format(config['dataset']), tsne=True)
            plot_scores_2d(X_embedded, Y.replace(replace_hyp).values, savefig = './plots/{}/tsne/target_'.format(config['dataset']), tsne=True)
            plot_scores_2d(X_embedded, Y.replace(replace_hyp).values, savefig='./plots/{}/tsne/target_'.format(config['dataset']), dim_1=1, dim_2=3, tsne=True)
            plot_scores_2d(X_embedded, Y.replace(replace_hyp).values, savefig='./plots/{}/tsne/target_'.format(config['dataset']), dim_1=2, dim_2=3, tsne=True)
            plot_density(X_embedded, Y.replace(replace_hyp).values, dim = 1, savefig = './plots/{}/tsne/target_'.format(config['dataset']), tsne=True)
            plot_density(X_embedded, Y.replace(replace_hyp).values, dim = 2, savefig='./plots/{}/tsne/target_'.format(config['dataset']), tsne=True)

    if config['plot_scores_colored_by_cluster'] is True and config['tsne'] is True:
        # PCA PLOTS AND T-SNE PLOTS COLORED BY CLUSTER
        pca, _ = PCA(X.values, config['num_dimensions'], savefig='./plots/{}/pca/'.format(config['dataset']), verbose=False)
        scores = pca.fit_transform()

        km = np.load('./results/km_{}.npy'.format(config['dataset']))
        agg = np.load('./results/agg_{}.npy'.format(config['dataset']))
        pca_km = np.load('./results/pca_km_{}.npy'.format(config['dataset']))
        pca_agg = np.load('./results/pca_agg_{}.npy'.format(config['dataset']))
        fa_km = np.load('./results/fa_km_{}.npy'.format(config['dataset']))
        fa_agg = np.load('./results/fa_agg_{}.npy'.format(config['dataset']))

        for (cluster, name) in [(km, 'km'), (agg, 'agg'), (pca_km, 'pca_km'), (pca_agg, 'pca_agg'), (fa_km, 'fa_km'), (fa_agg, 'fa_agg')]:
            plot_scores_2d(X_embedded, cluster, savefig='./plots/{}/tsne/{}_'.format(config['dataset'], name), tsne=True)
            plot_scores_2d(scores, cluster, savefig='./plots/{}/pca/{}_'.format(config['dataset'], name), tsne=False)
            plot_density(X_embedded, cluster, dim=1, savefig='./plots/{}/tsne/{}_'.format(config['dataset'], name), tsne=True)
            plot_density(scores, cluster, dim=1, savefig='./plots/{}/pca/{}_'.format(config['dataset'], name), tsne=False)

if __name__ == '__main__':
	main()