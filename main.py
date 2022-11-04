import argparse
from datasets import preprocess_vote, preprocess_hypothyroid
from dim_reduction import PCA
from sklearn.manifold import TSNE
from visualize import plot_scores_2d, plot_scores_3d, plot_density, plot_loadings
from kmeans import KMeans
from sklearn.cluster import AgglomerativeClustering
from utils import evaluate_clustering_number, make_plots
parser = argparse.ArgumentParser()

### run--> python main.py --dataset vote
parser.add_argument("--dataset", type=str, default='vote', choices=['vote', 'hyp'])
parser.add_argument("--dimReduction", type=str, default='agg', choices=['pca', 'fa'])
parser.add_argument("--num_dimensions", type=int, default=3)
parser.add_argument("--clusteringAlg", type=str, default='agg', choices=['km', 'agg'])
parser.add_argument("--num_clusters", default=2, type=int)
parser.add_argument("--max_num_clusters", type=int, default=7, choices=range(2,100))
parser.add_argument("--visualize_results", type=bool, default=False)
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
                'num_clusters': con.num_clusters,
                'affinity': con.affinity,
                'linkage': con.linkage,
                'max_num_clusters': con.max_num_clusters,
                'visualize_results': con.visualize_results
             }
    return config

def main():
    config = configuration()

    if config['visualize_results']:
        make_plots(config, metric = 'sil')
        make_plots(config, metric = 'ari')
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
        if config['dataset'] == 'hyp':
            cluster_no_dimred = KMeans(2).fit_predict(X.values)

    if config['clusteringAlg'] == 'agg':
        evaluate_clustering_number(config, X, Y)
        if config['dataset'] == 'vote':
            cluster_no_dimred = AgglomerativeClustering(n_clusters = 2, affinity='euclidean', linkage='complete').fit_predict(X.values)
        if config['dataset'] == 'hyp':
            cluster_no_dimred = AgglomerativeClustering(n_clusters = 2, affinity='euclidean', linkage='single').fit_predict(X.values)

    best_configs = {
        'vote':{'pca':{'kmeans':[3], 'agg':[2, 'complete']},
                'tsne':{'kmeans':[2],'agg':[2, 'ward']}},
        # revisar hyp pq no está bien
        'hyp': {'pca': {'kmeans': [3], 'agg': [2, 'complete']},
                 'tsne': {'kmeans': [2], 'agg': [2, 'ward']}}
    }

    # perform dimensionality reduction
    if config['dimReduction'] == 'pca':
        pca = PCA(X.values, config['num_dimensions'], savefig = './plots/{}/pca/'.format(config['dataset']), verbose = True)
        scores = pca.fit_transform()
        # perform clustering analysis
        evaluate_clustering_number(config, scores, Y, dim_reduc=True)
        if config['clusteringAlg'] == 'km':
            cluster_dimred = KMeans(best_configs[config['dataset']]['pca']['kmeans'][0]).fit_predict(scores)
        if config['clusteringAlg'] == 'agg':
            cluster_dimred = AgglomerativeClustering(n_clusters=best_configs[config['dataset']]['pca']['agg'][0],
                                                     affinity='euclidean',
                                                     linkage=best_configs[config['dataset']]['pca']['agg'][1]).fit_predict(scores)

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
        # plot scores colored by cluster
        plot_scores_2d(scores, cluster_no_dimred, savefig='./plots/{}/pca/{}_no_dimred_'.format(config['dataset'], config['clusteringAlg']))
        plot_scores_2d(scores, cluster_dimred, savefig='./plots/{}/pca/{}_dimred_'.format(config['dataset'], config['clusteringAlg']))
        # plot density colored by cluster
        plot_density(scores, cluster_no_dimred, dim=1, savefig='./plots/{}/pca/{}_no_dimred_'.format(config['dataset'], config['clusteringAlg']))
        plot_density(scores, cluster_dimred, dim=1, savefig='./plots/{}/pca/{}_dimred_'.format(config['dataset'], config['clusteringAlg']))

    if config['dimReduction'] == 'fa':
        pass

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

    # plot embeddings colored by cluster
    plot_scores_2d(X_embedded, cluster_no_dimred, savefig='./plots/{}/tsne/{}_no_dimred_'.format(config['dataset'], config['clusteringAlg']), tsne = True)
    plot_scores_2d(X_embedded, cluster_dimred, savefig='./plots/{}/tsne/{}_dimred_'.format(config['dataset'], config['clusteringAlg']), tsne = True)
    # plot density colored by cluster
    plot_density(X_embedded, cluster_no_dimred, dim=1, savefig='./plots/{}/tsne/{}_no_dimred_'.format(config['dataset'], config['clusteringAlg']), tsne = True)
    plot_density(X_embedded, cluster_dimred, dim=1, savefig='./plots/{}/tsne/{}_dimred_'.format(config['dataset'], config['clusteringAlg']), tsne = True)

if __name__ == '__main__':
	main()