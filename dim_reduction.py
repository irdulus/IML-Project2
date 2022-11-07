import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ncx2

class PCA:
    def __init__(self, X, d, savefig = None, verbose = False):
        """
        PCA object
        Parameters
        ----------

        X: nxp matrix
        d: number of principal components to be chosen
        """
        self.X = X
        self.d = d
        self.n = X.shape[0]
        self.c = X.shape[1]
        self.savefig = savefig
        self.verbose = verbose

    def __compute(self):

        self.mean_vector = self.X.mean(axis = 0, keepdims=True)
        self.cov_mat = np.cov(self.X.T)

        if self.verbose:
            print('-------------------')
            print('Covariance matrix:')
            print(self.cov_mat)
            print('###################')

        # eigenvalue/eigenvector decomposition of the covariance matrix
        self.eigval, self.eigvec = np.linalg.eig(self.cov_mat)

        if self.verbose:
            print('Eigenvectors:')
            print(self.eigvec)
            print('###################')

            print('Eigenvalues')
            print(self.eigval)
            print('-------------------')

        # order eigenvecors by eigenvalue
        tuples_eig = sorted([(self.eigval[i], self.eigvec[:, i]) for i in range(len(self.eigval))], reverse=True)
        if self.verbose:
            for eigenvalue, eigenvector in tuples_eig:
                print('Eigenvalue: ', eigenvalue)
                print('Eigenvector: ', eigenvector)
            print('-------------------')
        self.tuples_eig = tuples_eig
        self.tuples_eig_d = self.tuples_eig[:self.d]
        self.loadings = np.array([list(j[1]) for j in self.tuples_eig_d]).T

    def fit_transform(self):
        """
        Compute scores
        Output:

        scores: T = X * W
        """
        self.__compute()
        self.scores = (self.X - self.mean_vector) @ self.loadings

        ## eigenvalues plot
        self.__plot_eigenvalues(self.tuples_eig)
        n_com_90_ex_var = self.__plot_explained_variance(self.tuples_eig)
        ## reconstructed data matrix
        X_pred = self.__reconstruct(n_com_90_ex_var)
        error = self.X - X_pred
        scr = np.sum(error**2, axis = 1) # residual error sum of squares
        spe = np.sqrt(scr) # sum of predicted errors
        self.__plot_spe(scr, spe)
        return self.scores, n_com_90_ex_var

    def __reconstruct(self, n_com_90_ex_var):
        return (self.scores[:,:n_com_90_ex_var] @ self.loadings.T) + self.mean_vector

    def __plot_explained_variance(self, tuples_eig):
        total_variance = sum([i[0] for i in tuples_eig])
        var_explained = [(i[0] / total_variance) * 100 for i in tuples_eig]
        cumulative_var_exp = np.cumsum(var_explained)
        # select 90% explained variance
        n_com_90_ex_var = [index for index, v in enumerate(cumulative_var_exp) if v >= 90][0]
        plt.style.use('seaborn-whitegrid')
        fig = plt.figure(figsize=(9, 7))
        plt.bar(range(len(tuples_eig)), var_explained, alpha=0.85, align='center', label='Explained variance')
        plt.step(range(len(tuples_eig)), cumulative_var_exp, where='mid', label='Cumulative Explained variance')
        plt.axvline(n_com_90_ex_var, color='r')
        plt.axhline(90, color='r')
        plt.xticks(list(range(len(tuples_eig))))
        plt.xlabel('Principal components')
        plt.ylabel('Explained variance')
        plt.title('Explained variance')
        if self.savefig is not None:
            plt.savefig(self.savefig + 'explained_variance_pca.jpg', dpi = 350, bbox_inches='tight')
        plt.close()
        return n_com_90_ex_var
    def __plot_eigenvalues(self, tuples_eig):
        eigenvalues_plot = [i[0] for i in tuples_eig]
        plt.style.use('seaborn-whitegrid')
        fig = plt.figure(figsize=(9,7))
        principal_components = list(range(1, len(eigenvalues_plot) + 1))
        plt.bar(principal_components, eigenvalues_plot, alpha=0.85, align = 'center')
        plt.axhline(y = 1, color = 'r')
        plt.ylabel('Eigenvalues')
        plt.xlabel('Principal components')
        plt.xticks(principal_components)
        plt.title('Eigenvalues of the covariance matrix')
        if self.savefig is not None:
            plt.savefig(self.savefig + 'eigenvalues_pca.jpg', dpi = 350, bbox_inches='tight')
        plt.close()

    def __plot_spe(self, scr, spe):
        # chi2 confidence limit
        g = np.var(scr) / (2*np.mean(scr))
        h = (2*(np.mean(scr)**2))/np.var(scr)
        chi2lim = ncx2.ppf(0.99, h, g)
        # plot
        plt.style.use('seaborn-whitegrid')
        fig = plt.figure(figsize=(9, 7))
        plt.plot(range(spe.shape[0]), spe)
        plt.axhline(y=np.sqrt(chi2lim), color='r', label = '$\chi2$ lim')
        plt.ylabel('SPE')
        plt.xlabel('Observations')
        plt.title('Sum of predicted errors per observation')
        plt.legend()
        if self.savefig is not None:
            plt.savefig(self.savefig + 'spe_pca.jpg', dpi = 350, bbox_inches='tight')
        plt.close()