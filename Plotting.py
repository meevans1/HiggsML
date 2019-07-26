import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

def plot_hist(sig,bkg,xvar='x'):
    n_bins = 40
    
    d_min = min(sig[xvar].min(),bkg[xvar].min())
    d_max = max(sig[xvar].max(),bkg[xvar].max())
    if d_min<-998: d_min = 0
    sig_tr,bins,_ = plt.hist(bkg[xvar],bins=n_bins,range=(d_min,d_max), color='tab:orange', label='bkg',alpha=0.6, density=True)
    bkg_tr,_,_ = plt.hist(sig[xvar],bins=n_bins,range=(d_min,d_max), color='tab:blue', label='sig', alpha=0.6, density=True)

    plt.xlabel(xvar)
    
    plt.legend()
    plt.show()
    return

def plot_scatter(sig,bkg,xvar='x',yvar='y'):
    plt.plot(sig[xvar],sig[yvar], 'o', c='tab:blue', label='sig', alpha=0.5, markeredgecolor='k')
    plt.plot(bkg[xvar],bkg[yvar], 'o', c='tab:orange', label='bkg', alpha=0.5, markeredgecolor='k')
    plt.legend()
    #plt.show()  
    return

def make_meshgrid(x, y, h=.01):
    x_min, x_max = x.min() - 0.1, x.max() + 0.1
    y_min, y_max = y.min() - 0.1, y.max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = plt.contourf(xx, yy, Z, **params)
    return out


def plot_output(clf,sig_train,bkg_train,sig_test=None,bkg_test=None):

    n_bins = 40
    
    sig_train_output = clf.decision_function(sig_train.values)
    bkg_train_output = clf.decision_function(bkg_train.values)
    
    d_min = min(sig_train_output.min(),bkg_train_output.min())
    d_max = max(sig_train_output.max(),bkg_train_output.max())
    
    sig_tr,bins,_ = plt.hist(bkg_train_output,bins=n_bins,range=(d_min,d_max), color='tab:orange', label='bkg train',alpha=0.6, density=True)
    bkg_tr,_,_ = plt.hist(sig_train_output,bins=n_bins,range=(d_min,d_max), color='tab:blue', label='sig train', alpha=0.6, density=True)


    if sig_test != None and bkg_test != None:
        sig_test_output = clf.decision_function(sig_test.values)
        bkg_test_output = clf.decision_function(bkg_test.values)
        
        bin_centers = (bins[:-1]+bins[1:])/2
        sig_te,_ = np.histogram(sig_test_output,bins=bins,density=True)
        bkg_te,_ = np.histogram(bkg_test_output,bins=bins,density=True)

        plt.plot(bin_centers,bkg_te, 'o', c='tab:orange', label='bkg test', alpha=0.9, markeredgecolor='k')
        plt.plot(bin_centers,sig_te, 'o', c='tab:blue', label='sig test', alpha=0.9, markeredgecolor='k')

    #print(ks_2samp(sig_tr,sig_te))
    
    plt.legend()
    plt.show()

    return
