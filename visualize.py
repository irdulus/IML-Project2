import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_loadings(loadings, columns, savefig = None, dim_1=1, dim_2=2):
    colors = ['#689F38', '#039BE5', '#FF6F00', '#F44336', '#26C6DA']
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 10))
    for i, txt in enumerate(columns):
        ax.annotate(txt, (loadings[i, dim_1-1], loadings[i, dim_2-1]))
    plt.grid(True)
    plt.title('Loadings plot')

    plt.xlabel('Principal component {}'.format(dim_1))
    plt.ylabel('Principal component {}'.format(dim_2))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    if savefig is not None:
        plt.savefig(savefig + 'loadings_{}_{}.jpg'.format(dim_1, dim_2), dpi=350, bbox_inches='tight')
    plt.close()

def plot_scores_2d(scores, Y, savefig = None, dim_1=1, dim_2=2, tsne=False):
    colors = ['#689F38', '#039BE5', '#FF6F00', '#F44336', '#26C6DA']
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(15, 10))
    plt.axhline(c='black', alpha=0.2)
    plt.axvline(c='black', alpha=0.2)
    for i, group in enumerate(np.unique(Y)):
        plt.scatter(scores[np.where(Y == group),dim_1 - 1], scores[np.where(Y == group),dim_2 - 1], alpha = 0.82, c=colors[i], label = group)
    plt.grid(True)
    plt.title('PCA Space')
    plt.legend()
    plt.xlabel('Principal component {}'.format(dim_1) if tsne is False else 'Embedding {}'.format(dim_1))
    plt.ylabel('Principal component {}'.format(dim_2) if tsne is False else 'Embedding {}'.format(dim_2))
    if savefig is not None:
        plt.savefig(savefig + 'scores_{}_{}.jpg'.format(dim_1, dim_2), dpi=350, bbox_inches='tight')
    plt.close()

def plot_scores_3d(scores, Y, savefig = None, dim_1=1, dim_2=2, dim_3=3, tsne=False):
    colors = ['#689F38', '#039BE5', '#FF6F00', '#F44336', '#26C6DA', '#9C27B0', '#E64A19', '#EC407A', '#37474F',
              '#00796B', '#304FFE', '#6D4C41', '#AA00FF']
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(projection="3d")
    for i, group in enumerate(np.unique(Y)):
        ax.scatter3D(scores[np.where(Y == group),dim_1 - 1], scores[np.where(Y == group),dim_2 - 1], scores[np.where(Y == group),dim_3 - 1],
                    alpha = 0.82, c=colors[i], label = group)
    plt.grid(True)
    plt.title('PCA Space')
    plt.legend()
    ax.set_xlabel('Principal component {}'.format(dim_1) if tsne is False else 'Embedding {}'.format(dim_1))
    ax.set_ylabel('Principal component {}'.format(dim_2) if tsne is False else 'Embedding {}'.format(dim_2))
    ax.set_zlabel('Principal component {}'.format(dim_3) if tsne is False else 'Embedding {}'.format(dim_3))
    ax.view_init(10, 70)
    if savefig is not None:
        plt.savefig(savefig + 'scores3d_{}_{}.jpg'.format(dim_1, dim_2), dpi=350, bbox_inches='tight')
    plt.close()

def plot_density(scores, Y, savefig = None, dim=1, tsne=False):
    colors = ['#689F38', '#039BE5', '#FF6F00', '#F44336', '#26C6DA']
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(20, 8))
    plt.axhline(c='black', alpha=0.2)
    plt.axvline(c='black', alpha=0.2)
    for i, group in enumerate(np.unique(Y)):
        sns.distplot(scores[np.where(Y == group),dim-1], hist = False,
                         kde = True, kde_kws = {'linewidth': 3},  label=group)
    plt.grid(True)
    plt.title('Kernel Density Estimation (KDE) in the Principal Component {}'.format(dim) if tsne is False else 'Kernel Density Estimation (KDE) in Embedding {}'.format(dim))
    plt.legend()
    plt.xlabel('Principal component {}'.format(dim) if tsne is False else 'Embedding {}'.format(dim))
    if savefig is not None:
        plt.savefig(savefig + 'density_{}.jpg'.format(dim), dpi=350, bbox_inches='tight')
    plt.close()