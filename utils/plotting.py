import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def meanL1(gens, tars):
        """Compute MAE on test set for each of the fields"""
        tar_hist, bin_edges = np.histogram(tars[:,4,:,:,:], bins=50)
        gen_hist, _ = np.histogram(gens[:,4,:,:,:], bins=bin_edges)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        fig = plt.figure()
        plt.subplot(1,2,1)
        plt.errorbar(centers, tar_hist, yerr=np.sqrt(tar_hist), fmt='ks--', label='real')
        plt.errorbar(centers, gen_hist, yerr=np.sqrt(gen_hist), fmt='ro', label='generated')
        plt.xlabel('normalized T')
        plt.ylabel('Counts')
        plt.yscale('log')
        plt.legend()

        tar_hist, bin_edges = np.histogram(tars[:,0,:,:,:], bins=50)
        gen_hist, _ = np.histogram(gens[:,0,:,:,:], bins=bin_edges)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.subplot(1,2,2)
        plt.errorbar(centers, tar_hist, yerr=np.sqrt(tar_hist), fmt='ks--', label='real')
        plt.errorbar(centers, gen_hist, yerr=np.sqrt(gen_hist), fmt='ro', label='generated')
        plt.xlabel('normalized rho')
        plt.ylabel('Counts')
        plt.yscale('log')
        plt.legend()
        tar_clean = tar_hist[tar_hist >= 1.]
        gen_clean = gen_hist[tar_hist >= 1.]
        return (fig,
               np.sum(np.divide(np.power(tar_clean - gen_clean, 2.0), tar_clean)),
               np.mean(np.abs(tars - gens), axis=(0,2,3,4)),
               )


def generate_images(test_input, prediction, tar):
        """Visualize middle slice of the generated cubes"""
        names = [r'$\rho_N$', r'$vx_N$', r'$vy_N$', r'$vz_N$',
                 r'$\rho_H^*$', r'$vx_H^*$', r'$vy_H^*$', r'$vz_H^*$',
                 r'$\rho_H^* - \rho_H$', r'$T^*$', r'$T^* - T$',]
        cmaps = ['Blues', 'seismic', 'seismic', 'seismic',
                 'inferno', 'seismic', 'seismic', 'seismic',
                 'seismic', 'viridis', 'seismic']
        dat = [test_input, test_input, test_input, test_input,
               prediction, prediction, prediction, prediction,
               (prediction - tar), prediction, (prediction - tar)/tar]
        idxs = [0, 1, 2, 3, 0, 1, 2, 3, 0, 4, 4]
        norms = [(0, 0.9), (-1,1), (-1,1), (-1,1),
                 (-0.25, 0.92), (-1,1), (-1,1), (-1,1),
                 (-1., 1.), (-0.7, 0.7), (-1., 1.)]
        fig = plt.figure(figsize=(15,15))
        for i in range(11):
            plt.subplot(3,4,i+1)
            plt.imshow(dat[i][idxs[i],32,:,:], cmap=cmaps[i], norm=Normalize(vmin=norms[i][0], vmax=norms[i][1]))
            plt.axis('off')
            plt.title(names[i])
        return fig
