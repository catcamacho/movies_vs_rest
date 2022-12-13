import pandas as pd
import numpy as np
import nibabel as nib
import os
import scipy.stats as scp
from sklearn.svm import SVC, SVR
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, mean_squared_error
from sklearn.model_selection import cross_validate, cross_val_predict, GroupKFold, permutation_test_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import random
import numba as nb


def regress_covariates(data, covariates):
    """
    Parameters
    ----------
    data: numpy array
        a 1D or 2D dataset to regress the covariates from.
    covariates: numpy array
        a 1D or 2D array to regress from the data.
    
    Return
    ------
    resids: numpy array
        The residuals from the regression model of the same size as data.
    
    """
    if data.ndim==1:
        data = np.expand_dims(data, axis=1)
        
    if covariates.ndim==1:
        covariates = np.expand_dims(covariates, axis=1)
    
    coefficients = np.zeros((covariates.shape[1],data.shape[1]))
    resids = np.zeros(data.shape)

    # perform column-wise matrix inversion
    for x in range(0,data.shape[1]):
        y = data[:,x]
        inv_mat = np.linalg.pinv(covariates)
        coefficients[:,x] = np.dot(inv_mat,y)
        yhat = np.sum(np.transpose(coefficients[:,x])*covariates,axis=1)
        resids[:,x] = y - np.transpose(yhat)
    
    return(np.squeeze(resids))


def cv_fit(model, X, Y, cv, groups=None):
    """
    This function fits a support vetor model, with cross validation.
    
    Parameters
    ----------
    model: sklearn SVC or SVR object
        The model being trained
    X: numpy array
        The feature set being trained of size Nsamples x Nmeasures
    Y: numpy array
        The labels the model is being trained to predict from X. Must be size Nsamples
    groups: numpy array
        Optional. If there are multiple samples from the same participant, Subject ID 
        should be passed here.  Array must be size Nsamples.
    
    Returns
    -------
    estimators: list
        
    weights: np.array
        
    mean_weights: np.array
        
    Y_pred: np.array
        
    train_scores: np.array
        
    """
    
    if isinstance(model, SVC):
        scoring = 'accuracy'
    elif isinstance(model, SVR):
        scoring = 'neg_mean_squared_error'
    results = cross_validate(model, X=X, y=Y, groups=groups, n_jobs=10,
                       cv=cv, return_estimator=True, scoring=scoring)
    Y_pred = cross_val_predict(model, X=X, y=Y, groups=groups, n_jobs=10, cv=cv)
    train_scores = results['test_score']

    for i, a in enumerate(results['estimator']):
        c = np.expand_dims(a.coef_, axis=2)
        if i==0:
            weights = c
        else:
            weights = np.concatenate([weights, c], axis=2)
    
    estimators = results['estimator']
    weights = np.absolute(weights)
    mean_weights = np.mean(np.mean(weights, axis=2), axis=0, keepdims=True)
    return(estimators, weights, mean_weights, Y_pred, train_scores)


def predict_out(X, Y, estimators, kind):
    """
    This function takes a trained model (fit with cross-validation) and applies to the a separate 
    dataset. The consistency between actual and predicted labels is used to benchmark model accuracy.
    
    Parameters
    ----------
    X: numpy array
        The feature set being used to test the model on of size Nsamples x Nmeasures
    Y: numpy array
        The labels the model is supposed to guess from X. Must be size Nsamples
    estimators: list
        The list of fitted models from cross-validation
    kind: str ['classifier', 'regress']
        Indicates the type of support vector model used.
    
    Returns
    -------
    estimators: 
    weights: 
    mean_weights: 
    Y_pred: 
    train_scores: 
    """
    index = range(0,len(Y))
    for i, a in enumerate(estimators):
        if i==0:
            Y_pred = a.predict(X)
            Y_test = Y
            ind = index
        else:
            Y_pred = np.concatenate([Y_pred, a.predict(X)], axis=0)
            Y_test = np.concatenate([Y_test, Y], axis=0)
            ind = np.concatenate([ind, index], axis=0)

    if kind=='classifier':
        accuracy = pd.DataFrame.from_dict(classification_report(Y_test, Y_pred, output_dict=True)).T
    elif kind=='regress':
        var_series = pd.Series(Y_pred, index=ind)
        var_series = var_series.groupby(var_series.index).mean()
        accuracy = pd.DataFrame(columns = ['stat','pval'])
        Y_test = Y
        Y_pred = var_series.to_numpy()
        accuracy.loc['SpearmanR','stat'], accuracy.loc['SpearmanR','pval'] = scp.spearmanr(Y_pred, Y_test)
        accuracy.loc['PearsonR','stat'], accuracy.loc['PearsonR','pval'] = scp.pearsonr(Y_pred, Y_test)
        slope, intercept, r, p, se = scp.linregress(Y_pred, Y_test)
        accuracy.loc['LinearB','stat'] = slope
        accuracy.loc['LinearB','pval'] = p
        accuracy.loc['MSE','stat'] = mean_squared_error(Y_pred, Y_test)
    return(Y_pred, accuracy)


def make_consistency_plot(Y, Y_pred, folds, outfile_name):
    if Y.shape[0] < Y_pred.shape[0]:
        Y_orig = Y
        for i in range(1, folds):
            Y = np.concatenate([Y,Y_orig], axis=0)
            
    if Y.shape[0] == Y_pred.shape[0]:
        plt.figure(figsize=(4,3))
        plt.scatter(Y, Y_pred)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.tight_layout()
        plt.savefig(outfile_name)
        plt.show()
        plt.close()
    else:
        print("ERROR: length mismatch")  
        
        
def boot_predict(estimators, X, Y, outdir, kind='regress', ci=95, samples=10000):
    """
    
    """
    
    # determine percentiles for the CI estimation from the bootstrapped distribution
    lower = (100 - ci)/2
    upper = 100 - lower

    if kind=='classifier':
        scoring = 'accuracy'
    elif kind=='regress':
        scoring = 'neg_mean_squared_error'

    test_scores = []
    pearsonr = []
    spearmanr = []

    for a in nb.prange(samples):
        bootsample_size = random.randint(int(len(Y)*0.5),int(len(Y)*0.75))
        subsampmask = np.full(len(Y), 0)
        subsampmask[:bootsample_size] = 1
        np.random.shuffle(subsampmask)
        X_temp = X[subsampmask==1,:]
        Y_temp = Y[subsampmask==1]

        for a in estimators:
            Y_pred = a.predict(X_temp)
            if kind=='classifier':
                t = classification_report(Y_temp, Y_pred, output_dict=True)
                test_scores.append(t['accuracy'])
            elif kind=='regress':
                mse = mean_squared_error(Y_temp, Y_pred)
                r, p = scp.spearmanr(Y_temp, Y_pred)
                test_scores.append(mse)
                spearmanr.append(r)
                r, p = scp.pearsonr(Y_temp, Y_pred)
                pearsonr.append(r)

    # test if boot strapped distibution is normally distributed
    test_scores = np.array(test_scores)
    k , p = scp.kstest(test_scores, 'norm')        

    # store and save results
    results = pd.DataFrame(columns = ['boot_mean','boot_SD','boot_median','KSstat','KSpval','CI','lowerCI','upperCI'])
    results.loc['test_scores','boot_mean'] = np.mean(test_scores)
    results.loc['test_scores','boot_SD'] = np.std(test_scores)
    results.loc['test_scores','boot_median'] = np.percentile(test_scores, 50)
    results.loc['test_scores','CI'] = ci
    results.loc['test_scores','lowerCI'] = np.percentile(test_scores, lower)
    results.loc['test_scores','upperCI'] = np.percentile(test_scores, upper)
    results.loc['test_scores','KSstat'] = k
    results.loc['test_scores','KSpval'] = p

    if kind=='regress':
        k , p = scp.kstest(spearmanr, 'norm')
        spearmanr = np.array(spearmanr)

        results.loc['spearmanr','boot_mean'] = np.mean(spearmanr)
        results.loc['spearmanr','boot_SD'] = np.std(spearmanr)
        results.loc['spearmanr','boot_median'] = np.percentile(spearmanr, 50)
        results.loc['spearmanr','CI'] = ci
        results.loc['spearmanr','lowerCI'] = np.percentile(spearmanr, lower)
        results.loc['spearmanr','upperCI'] = np.percentile(spearmanr, upper)
        results.loc['spearmanr','KSstat'] = k
        results.loc['spearmanr','KSpval'] = p

        k , p = scp.kstest(pearsonr, 'norm')
        pearsonr = np.array(pearsonr)

        results.loc['pearsonr','boot_mean'] = np.mean(pearsonr)
        results.loc['pearsonr','boot_SD'] = np.std(pearsonr)
        results.loc['pearsonr','boot_median'] = np.percentile(pearsonr, 50)
        results.loc['pearsonr','CI'] = ci
        results.loc['pearsonr','lowerCI'] = np.percentile(pearsonr, lower)
        results.loc['pearsonr','upperCI'] = np.percentile(pearsonr, upper)
        results.loc['pearsonr','KSstat'] = k
        results.loc['pearsonr','KSpval'] = p

    results.to_csv(os.path.join(outdir, 'bootstrapped_test_accuracy_randN.csv'))
    np.save(os.path.join(outdir, 'bootstrapped_distribution_testscores.npy'), test_scores)
    np.save(os.path.join(outdir, 'bootstrapped_distribution_spearmanr.npy'), spearmanr)
    np.save(os.path.join(outdir, 'bootstrapped_distribution_pearsonr.npy'), pearsonr)
    return(results)


def permuted_p(model, X, Y, cv, out_folder, train_score, test_score, groups=None, n_perms=1000):

    # Perform permutation testing to get a p-value
    if isinstance(model, SVC):
        scoring = 'accuracy'

        train_score, permutation_scores, pvalue = permutation_test_score(model, X, Y, scoring=scoring, 
                                                                         cv=cv, n_permutations=n_perms, n_jobs=10, groups=groups)
    elif isinstance(model, SVR):
        scoring = 'neg_mean_squared_error'
        if isinstance(Y, pd.Series):
            Y = Y.to_numpy()

        Y_shuff = Y
        scores = np.zeros((n_perms, cv))
        for a in range(0,n_perms):
            np.random.shuffle(Y_shuff)
            res = cross_validate(model, X=X, y=Y_shuff, groups=groups, n_jobs=10, cv=cv, scoring=scoring)
            scores[a,:] = res['test_score']

        permutation_scores = scores.flatten()

    # Save a figure of the permutation scores
    fig, ax = plt.subplots(figsize=(8,6))
    ax.hist(permutation_scores, 20, label='Permutation scores', density=True)
    ax.axvline(train_score, ls='-', color='m', label='Train')
    ax.axvline(test_score, ls='--', color='g', label='Test')
    if isinstance(model, SVC):
        ax.axvline(1. / len(Y_train.unique()), ls='--', color='k', label='Chance')   
    plt.legend()
    plt.xlabel('Score')
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, 'permutation_plot.svg'), transparent=True)
    plt.close()

    # Save scores 
    results = pd.DataFrame()
    results.loc['Train_Score','Stat'] = train_score
    results.loc['Test_Score','Stat'] = test_score
    results.loc['Train_Score','PermPval'] = (np.sum((permutation_scores>=train_score).astype(int)) + 1) / (n_perms*cv + 1)
    results.loc['Test_Score','PermPval'] = (np.sum((permutation_scores>=test_score).astype(int)) + 1) / (n_perms*cv + 1)
    
    results.to_csv(os.path.join(out_folder, 'permutation_stats.csv'))
    np.save(os.path.join(out_folder, 'permutation_score_distribution.npy'), permutation_scores)
    
    return(results)


def permuted_importance(estimators_list, X, Y, labels, out_folder, model_score, percent=20, n_perms=100):
    # determine which connections to permute
    weights = []
    for a in estimators_list:
        weights.append(np.abs(estimators_list[0].coef_))
    weights = np.mean(np.concatenate(weights, axis=0), axis=0)
    cutoff = np.percentile(weights, 100-percent)
    features_to_perm = weights>cutoff
    permed_feature_labels = np.array(labels)[features_to_perm]

    # set up permuation
    if isinstance(estimators_list[0], SVC):
        scoring = 'accuracy'
    elif isinstance(estimators_list[0], SVR):
        scoring = 'neg_mean_squared_error'

    np.save(os.path.join(out_folder, 'temp.npy'), X)
    rng = np.random.default_rng()
    perm_feat_imp = pd.DataFrame(index=permed_feature_labels, columns=range(0,n_perms))

    # conduct permutation
    for i in range(0,n_perms):
        for f in permed_feature_labels:
            print(f)
            # permute only the selected features
            perm_X = np.load(os.path.join(out_folder, 'temp.npy'))
            perm = perm_X[:,labels==f]
            perm = rng.permutation(rng.permutation(perm, axis=0), axis=1)
            perm_X[:,labels==f] = perm
            results = cross_validate(model, X=perm_X, y=Y, n_jobs=10, cv=cv, scoring=scoring)
            perm_imp = model_score - np.mean(results['test_score'])
            perm_feat_imp.loc[f,i] = perm_imp

    os.remove(os.path.join(out_folder, 'temp.npy'))

    # save dataframe with all the scores
    perm_feat_imp.to_csv(os.path.join(out_folder, 'permuted_importance_scores.csv'))

    # save dataframe with mean scores
    imp_mean=np.mean(perm_feat_imp, axis=0)
    imp_table = pd.DataFrame(index=labels)
    imp_table['mean_importance'] = np.squeeze(imp_mean)
    imp_table.to_csv(os.path.join(out_folder, 'mean_importance.csv'))

    # plot importance scores
    imp_table['feature'] = imp_table.index
    sorted_imp_table = imp_table.sort_values(by='mean_importance', axis=0, ascending=False)
    plt.figure(figsize=(6,5))
    sns.barplot(x='mean_importance', y='feature', data=sorted_imp_table.iloc[:20,:], ci=None, color="#3B75AF")
    plt.xlabel('Change in {0}'.format(scoring))
    plt.ylabel('')
    plt.axvline(0, color='gray', clip_on=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, 'top_20_important_features.svg'))
    
    
    