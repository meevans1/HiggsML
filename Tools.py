import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import Plotting

def evaluate_mva(clf,sig,bkg,xx=None,yy=None):
    X = np.concatenate( [sig.values,bkg.values] )
    y = np.concatenate( [np.ones(len(sig.index)),np.zeros(len(bkg.index))] )

    if xx == None and yy == None:
        X0, X1 = X[:,0], X[:,1]
        xx, yy = Plotting.make_meshgrid(X0,X1)

    Plotting.plot_contours(clf, xx, yy, colors=['tab:orange','tab:blue','tab:blue','tab:orange'], alpha=0.8)
    Plotting.plot_scatter(sig,bkg,'x','y')
    plt.show()
    return xx,yy


def train_mva(clf,sig,bkg,sig_weights,bkg_weights):
    X = np.concatenate( [sig.values,bkg.values] )
    y = np.concatenate( [np.ones(len(sig.index)),np.zeros(len(bkg.index))] )
    sample_weights = np.concatenate( [sig_weights,bkg_weights] )

    clf.fit(X,y,sample_weight=sample_weights)
    return

def calc_roc(clf,sig,bkg):
    X = np.concatenate( [sig.values,bkg.values] )
    y = np.concatenate( [np.ones(len(sig.index)),np.zeros(len(bkg.index))] )

    y_score = clf.decision_function(X)

    fpr, tpr, _ = roc_curve(y.ravel(), y_score.ravel())

    roc_auc = auc(fpr,tpr)

    plt.figure()
    plt.plot(tpr,1-fpr,color='tab:orange', lw=2, label='ROC Curve (area = %0.2f)'%roc_auc)
    #plt.plot([1,0],[1,0], color='tab:blue', lw=2, linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('Sig Eff')
    plt.ylabel('1 - Bkg Eff')
    plt.legend(loc='best')
    plt.show()
    
    return roc_auc
