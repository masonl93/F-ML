'''
Functions from Lopez de Prado's 'Advances in Financial Machine Learning'

Number above the functions refer to the book's code snippet.

Various modifications have been made:
    - Some functions have been enhanced as part of certain chapter's
      exercises
    - Python 3 compatibility
    - Fixing pandas, etc warnings/errors
    - Cleaner formatting
'''

import datetime as dt
from itertools import product
import multiprocessing as mp
from pathlib import PurePath, Path
from random import gauss
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
import scipy.stats as stats
from scipy.stats import rv_continuous, kstest
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection._split import _BaseKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


# 2.4
def getTEvents(gRaw, h):
    tEvents, sPos, sNeg = [], 0, 0
    diff = gRaw.diff()
    for i in diff.index[1:]:
        sPos, sNeg = max(0, sPos+diff.loc[i]), min(0, sNeg+diff.loc[i])
        if sNeg < -h:
            sNeg = 0
            tEvents.append(i)
        elif sPos > h:
            sPos = 0
            tEvents.append(i)
    return pd.DatetimeIndex(tEvents)


# 3.1
def getDailyVol(close, span0=100):
    # daily vol, reindexed to close
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(
        close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:])
    df0 = close.loc[df0.index]/close.loc[df0.values].values-1  # daily returns
    df0 = df0.ewm(span=span0).std()
    return df0


# 3.2
def applyPtSlOnT1(close, events, ptSl, molecule):
    # apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)
    if ptSl[0] > 0:
        pt = ptSl[0]*events_['trgt']
    else:
        pt = pd.Series(index=events.index)  # NaNs
    if ptSl[1] > 0:
        sl = -ptSl[1]*events_['trgt']
    else:
        sl = pd.Series(index=events.index)  # NaNs
    for loc, t1 in events_['t1'].fillna(close.index[-1]).iteritems():
        df0 = close[loc:t1]  # path prices
        df0 = (df0/close[loc]-1)*events_.at[loc, 'side']  # path returns
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]
                                 ].index.min()  # earliest stop loss
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]
                                 ].index.min()  # earliest profit taking
    return out


# 3.6
def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False, side=None):
    # 1) get target
    trgt = trgt.reindex(tEvents)
    trgt = trgt[trgt > minRet]  # minRet
    # 2) get t1 (max holding period)
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=tEvents)
    # 3) form events object, apply stop loss on t1
    if side is None:
        side_, ptSl_ = pd.Series(1., index=trgt.index), [ptSl[0], ptSl[0]]
    else:
        side_, ptSl_ = side.reindex(trgt.index), ptSl[:2]
    events = (pd.concat({'t1': t1, 'trgt': trgt,
                         'side': side_}, axis=1).dropna(subset=['trgt']))
    df0 = mpPandasObj(func=applyPtSlOnT1, pdObj=('molecule', events.index),
                      numThreads=numThreads, close=close, events=events,
                      ptSl=ptSl_)
    events['t1'] = df0.dropna(how='all').min(axis=1)  # pd.min ignores nan
    if side is None:
        events = events.drop('side', axis=1)
    return events


# 3.4
def addVerticalBarrier(tEvents, close, numDays=1):
    t1 = close.index.searchsorted(tEvents + pd.Timedelta(days=numDays))
    t1 = t1[t1 < close.shape[0]]
    # NaNs at end
    t1 = (pd.Series(close.index[t1], index=tEvents[:t1.shape[0]]))
    return t1


# 3.7
def getBins(events, close):
    '''
    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    -events.index is event's starttime
    -events['t1'] is event's endtime
    -events['trgt'] is event's target
    -events['side'] (optional) implies the algo's position side
    Case 1: ('side' not in events): bin in (-1,1) <-label by price action
    Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    '''
    # 1) prices aligned with events
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    # 2) create out object
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values/px.loc[events_.index]-1
    if 'side' in events_:
        out['ret'] *= events_['side']  # meta-labeling
    out['bin'] = np.sign(out['ret'])
    if 'side' in events_:
        out.loc[out['ret'] <= 0, 'bin'] = 0  # meta-labeling
    return out


# Modifying getBins to return a 0 when vertical barrier is touched first
def getBins_v2(events, close, t1=None):
    '''
    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    -events.index is event's starttime
    -events['t1'] is event's endtime
    -events['trgt'] is event's target
    -events['side'] (optional) implies the algo's position side
    -t1 is vertical barrier series
    Case 1: ('side' not in events): bin in (-1,1) <-label by price action
    Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    '''
    # 1) prices aligned with events
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    # 2) create out object
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values/px.loc[events_.index]-1
    if 'side' in events_:
        out['ret'] *= events_['side']  # meta-labeling
    out['bin'] = np.sign(out['ret'])
    if 'side' in events_:
        out.loc[out['ret'] <= 0, 'bin'] = 0  # meta-labeling
    # ADDED VERTICAL BARRIER CODE HERE
    elif t1 is not None:
        # Not applicable to meta-labeling
        # Finding all timestamps of hitting a barrier (endtime)
        # which is also in our vertical barrier series (t1)
        vertical_touches = events[events['t1'].isin(t1.values)].index
        out.loc[vertical_touches, 'bin'] = 0
    # END
    return out


# 3.8
def dropLabels(events, minPct=.05):
    # apply weights, drop labels with insufficient examples
    while True:
        df0 = events['bin'].value_counts(normalize=True)
        if df0.min() > minPct or df0.shape[0] < 3:
            break
        print('dropped label: ', df0.argmin(), df0.min())
        events = events[events['bin'] != df0.argmin()]
    return events


# 4.1
def mpNumCoEvents(closeIdx, t1, molecule):
    '''
    Compute the number of concurrent events per bar.
    +molecule[0] is the date of the first event on which the weight will be computed
    +molecule[-1] is the date of the last event on which the weight will be computed
    Any event that starts before t1[modelcule].max() impacts the count.
    '''
    # 1) find events that span the period [molecule[0],molecule[-1]]
    # unclosed events still must impact other weights
    t1 = t1.fillna(closeIdx[-1])
    t1 = t1[t1 >= molecule[0]]  # events that end at or after molecule[0]
    # events that start at or before t1[molecule].max()
    t1 = t1.loc[:t1[molecule].max()]
    # 2) count events spanning a bar
    iloc = closeIdx.searchsorted(np.array([t1.index[0], t1.max()]))
    count = pd.Series(0, index=closeIdx[iloc[0]:iloc[1]+1])
    for tIn, tOut in t1.iteritems():
        count.loc[tIn:tOut] += 1.
    return count.loc[molecule[0]:t1[molecule].max()]


# 4.2
def mpSampleTW(t1, numCoEvents, molecule):
    # Derive avg. uniqueness over the events lifespan
    wght = pd.Series(index=molecule)
    for tIn, tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn] = (1./numCoEvents.loc[tIn:tOut]).mean()
    return wght


# 4.3
def getIndMatrix(barIx, t1):
    # Get indicator matrix
    indM = pd.DataFrame(0, index=barIx, columns=range(t1.shape[0]))
    for i, (t0, t1) in enumerate(t1.iteritems()):
        indM.loc[t0:t1, i] = 1.
    return indM


# 4.4
def getAvgUniqueness(indM):
    # Average uniqueness from indicator matrix
    c = indM.sum(axis=1)  # concurrency
    u = indM.div(c, axis=0)  # uniqueness
    avgU = u[u > 0].mean()  # average uniqueness
    return avgU


# 4.5
def seqBootstrap(indM, sLength=None):
    # Generate a sample via sequential bootstrap
    if sLength is None:
        sLength = indM.shape[1]
    phi = []
    while len(phi) < sLength:
        avgU = pd.Series()
        for i in indM:
            indM_ = indM[phi+[i]]  # reduce indM
            avgU.loc[i] = getAvgUniqueness(indM_).iloc[-1]
        prob = avgU/avgU.sum()  # draw prob
        phi += [np.random.choice(indM.columns, p=prob)]
    return phi


# 4.11
def getTimeDecay(tW, clfLastW=1.):
    # apply piecewise-linear decay to observed uniqueness (tW)
    # newest observation gets weight=1, oldest observation gets weight=clfLastW
    clfW = tW.sort_index().cumsum()
    if clfLastW >= 0:
        slope = (1.-clfLastW)/clfW.iloc[-1]
    else:
        slope = 1./((clfLastW+1)*clfW.iloc[-1])
    const = 1.-slope*clfW.iloc[-1]
    clfW = const+slope*clfW
    clfW[clfW < 0] = 0
    print(const, slope)
    return clfW


# 5.1
def getWeights(d, size):
    # thres>0 drops insignificant weights
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1]/k*(d-k+1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


# 5.1 cont.
def plotWeights(dRange, nPlots, size):
    w = pd.DataFrame()
    for d in np.linspace(dRange[0], dRange[1], nPlots):
        w_ = getWeights(d, size=size)
        w_ = pd.DataFrame(w_, index=range(w_.shape[0])[::-1], columns=[d])
        w = w.join(w_, how='outer')
    ax = w.plot()
    ax.legend(loc='upper left')
    plt.show()
    return


# 5.2
def fracDiff(series, d, thres=.01):
    '''
    Increasing width window, with treatment of NaNs
    Note 1: For thres=1, nothing is skipped.
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    '''
    # import pdb
    # pdb.set_trace()
    # 1) Compute weights for the longest series
    w = getWeights(d, series.shape[0])
    # 2) Determine initial calcs to be skipped based on weight-loss threshold
    w_ = np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = w_[w_ > thres].shape[0]
    print("skip: %f" % skip)
    # 3) Apply weights to values
    df = {}
    for name in series.columns:
        seriesF, df_ = series[[name]].fillna(
            method='ffill').dropna(), pd.Series()
        for iloc in range(skip, seriesF.shape[0]):
            loc = seriesF.index[iloc]
            if not np.isfinite(series.loc[loc, name]):
                continue  # exclude NAs
            df_.loc[loc] = np.dot(w[-(iloc+1):, :].T,
                                  seriesF.loc[:loc])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


# Pg. 82 - Fixed-Width WIndow Fracdiff formula
def getWeights_FFD(d, size, thres):
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1]/k*(d-k+1)
        if abs(w_) < thres:
            break  # not set w_ to 0?
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


# 5.3
def fracDiff_FFD(series, d, thres=1e-5):
    '''
    Constant width window (new solution)
    Note 1: thres determines the cut-off weight for the window
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    '''
    # import pdb
    # pdb.set_trace()
    # 1) Compute weights for the longest series
    w = getWeights_FFD(d, series.shape[0], thres)
    width = len(w)-1
    # 2) Apply weights to values
    df = {}
    for name in series.columns:
        seriesF, df_ = series[[name]].fillna(
            method='ffill').dropna(), pd.Series()
        for iloc1 in range(width, seriesF.shape[0]):
            loc0, loc1 = seriesF.index[iloc1-width], seriesF.index[iloc1]
            # print(np.isfinite(series.loc[loc1, name]))
            if not np.isfinite(series.loc[loc1, name]).any():
                continue  # exclude NAs
            df_.loc[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


# 7.3
class PurgedKFold(_BaseKFold):
    '''
    Extend KFold class to work with labels that span intervals
    The train is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o training samples in between
    '''

    def __init__(self, n_splits=3, t1=None, pctEmbargo=0.):
        if not isinstance(t1, pd.Series):
            raise ValueError('Label Through Dates must be a pd.Series')
        super().__init__(
            n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pctEmbargo = pctEmbargo

    def split(self, X, y=None, groups=None):
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError('X and ThruDateValues must have the same index')
        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0]*self.pctEmbargo)
        test_starts = [(i[0], i[-1]+1) for i in
                       np.array_split(np.arange(X.shape[0]), self.n_splits)]
        for i, j in test_starts:
            t0 = self.t1.index[i]  # start of test set
            test_indices = indices[i:j]
            maxT1Idx = self.t1.index.searchsorted(self.t1[test_indices].max())
            train_indices = self.t1.index.searchsorted(
                self.t1[self.t1 <= t0].index)
            if maxT1Idx < X.shape[0]:  # right train (with embargo)
                train_indices = np.concatenate(
                    (train_indices, indices[maxT1Idx+mbrg:]))
            yield train_indices, test_indices


# 7.4
def cvScore(clf, X, y, sample_weight, scoring='neg_log_loss', t1=None, cv=None, cvGen=None,
            pctEmbargo=None):
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise Exception('wrong scoring method.')
    if cvGen is None:
        cvGen = PurgedKFold(n_splits=cv, t1=t1,
                            pctEmbargo=pctEmbargo)  # purged
    score = []
    for train, test in cvGen.split(X=X):
        fit = clf.fit(X=X.iloc[train, :], y=y.iloc[train],
                      sample_weight=sample_weight.iloc[train].values)
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X.iloc[test, :])
            score_ = -log_loss(y.iloc[test], prob,
                               sample_weight=sample_weight.iloc[test].values, labels=clf.classes_)
        else:
            pred = fit.predict(X.iloc[test, :])
            score_ = accuracy_score(
                y.iloc[test], pred, sample_weight=sample_weight.iloc[test].values)
        score.append(score_)
    return np.array(score)


# 8.2
def featImpMDI(fit, featNames):
    # feat importance based on IS mean impurity reduction
    df0 = {i: tree.feature_importances_ for i,
           tree in enumerate(fit.estimators_)}
    df0 = pd.DataFrame.from_dict(df0, orient='index')
    df0.columns = featNames
    df0 = df0.replace(0, np.nan)  # because max_features=1
    imp = pd.concat({'mean': df0.mean(), 'std': df0.std()
                     * df0.shape[0]**-.5}, axis=1)
    imp /= imp['mean'].sum()
    return imp


# 8.3
def featImpMDA(clf, X, y, cv, sample_weight, t1, pctEmbargo, scoring='neg_log_loss'):
    # feat importance based on OOS score reduction
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise Exception('wrong scoring method.')
    cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)  # purged cv
    scr0, scr1 = pd.Series(), pd.DataFrame(columns=X.columns)
    for i, (train, test) in enumerate(cvGen.split(X=X)):
        X0, y0, w0 = X.iloc[train, :], y.iloc[train], sample_weight.iloc[train]
        X1, y1, w1 = X.iloc[test, :], y.iloc[test], sample_weight.iloc[test]
        fit = clf.fit(X=X0, y=y0, sample_weight=w0.values)
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X1)
            scr0.loc[i] = -log_loss(y1, prob, sample_weight=w1.values,
                                    labels=clf.classes_)
        else:
            pred = fit.predict(X1)
            scr0.loc[i] = accuracy_score(y1, pred, sample_weight=w1.values)
        for j in X.columns:
            X1_ = X1.copy(deep=True)
            np.random.shuffle(X1_[j].values)  # permutation of a single column
            if scoring == 'neg_log_loss':
                prob = fit.predict_proba(X1_)
                scr1.loc[i, j] = -log_loss(y1, prob, sample_weight=w1.values,
                                           labels=clf.classes_)
            else:
                pred = fit.predict(X1_)
                scr1.loc[i, j] = accuracy_score(
                    y1, pred, sample_weight=w1.values)
    imp = (-scr1).add(scr0, axis=0)
    if scoring == 'neg_log_loss':
        imp = imp/-scr1
    else:
        imp = imp/(1.-scr1)
    imp = pd.concat({'mean': imp.mean(), 'std': imp.std()
                     * imp.shape[0]**-.5}, axis=1)
    return imp, scr0.mean()


# 8.4
def auxFeatImpSFI(featNames, clf, trnsX, cont, scoring, cvGen):
    imp = pd.DataFrame(columns=['mean', 'std'])
    for featName in featNames:
        df0 = cvScore(clf, X=trnsX[[featName]], y=cont['bin'], sample_weight=cont['w'],
                      scoring=scoring, cvGen=cvGen)
        imp.loc[featName, 'mean'] = df0.mean()
        imp.loc[featName, 'std'] = df0.std()*df0.shape[0]**-.5
    return imp


# 8.5
def get_eVec(dot, varThres):
    # compute eVec from dot prod matrix, reduce dimension
    eVal, eVec = np.linalg.eigh(dot)
    idx = eVal.argsort()[::-1]  # arguments for sorting eVal desc
    eVal, eVec = eVal[idx], eVec[:, idx]
    # 2) only positive eVals
    eVal = pd.Series(eVal, index=['PC_'+str(i+1)
                                  for i in range(eVal.shape[0])])
    eVec = pd.DataFrame(eVec, index=dot.index, columns=eVal.index)
    eVec = eVec.loc[:, eVal.index]
    # 3) reduce dimension, form PCs
    cumVar = eVal.cumsum()/eVal.sum()
    dim = cumVar.values.searchsorted(varThres)
    eVal, eVec = eVal.iloc[:dim+1], eVec.iloc[:, :dim+1]
    return eVal, eVec


# 8.5 cont.
def orthoFeats(dfX, varThres=.95):
    # Given a dataframe dfX of features, compute orthofeatures dfP
    dfZ = dfX.sub(dfX.mean(), axis=1).div(dfX.std(), axis=1)  # standardize
    dot = pd.DataFrame(np.dot(dfZ.T, dfZ),
                       index=dfX.columns, columns=dfX.columns)
    eVal, eVec = get_eVec(dot, varThres)
    dfP = pd.DataFrame(np.dot(dfZ, eVec), index=dfZ.index,
                       columns=eVec.columns)
    return dfP


# 8.7
def getTestData(n_features=40, n_informative=10, n_redundant=10, n_samples=10000):
    # generate a random dataset for a classification problem
    trnsX, cont = make_classification(n_samples=n_samples, n_features=n_features,
                                      n_informative=n_informative, n_redundant=n_redundant, random_state=0,
                                      shuffle=False)
    try:
        df0 = pd.DatetimeIndex(periods=n_samples, freq=pd.tseries.offsets.BDay(),
                               end=pd.datetime.today())
    except OverflowError:
        # mlevy - fix deprecated error + handles dealing with overflow e.g. n_samples=1e6
        df0 = pd.DatetimeIndex(data=pd.date_range(
            end=pd.datetime.today(), periods=1e6, freq='1H'))
    trnsX, cont = pd.DataFrame(trnsX, index=df0), pd.Series(
        cont, index=df0).to_frame('bin')
    df0 = (['I_'+str(i) for i in range(n_informative)] +
           ['R_'+str(i) for i in range(n_redundant)])
    df0 += ['N_'+str(i) for i in range(n_features-len(df0))]
    trnsX.columns = df0
    cont['w'] = 1./cont.shape[0]
    cont['t1'] = pd.Series(cont.index, index=cont.index)
    return trnsX, cont


# 8.8
def featImportance(trnsX, cont, clf=None, n_estimators=1000, cv=10, max_samples=1., numThreads=24,
                   pctEmbargo=0, scoring='accuracy', method='SFI', minWLeaf=0., **kargs):
    # feature importance from a random forest
    # run 1 thread with ht_helper in dirac1
    n_jobs = (-1 if numThreads > 1 else 1)
    # 1) prepare classifier,cv. max_features=1, to prevent masking
    if clf is None:
        clf = DecisionTreeClassifier(criterion='entropy', max_features=1,
                                     class_weight='balanced', min_weight_fraction_leaf=minWLeaf)
        clf = BaggingClassifier(base_estimator=clf, n_estimators=n_estimators,
                                max_features=1., max_samples=max_samples, oob_score=True, n_jobs=n_jobs)
    fit = clf.fit(X=trnsX, y=cont['bin'], sample_weight=cont['w'].values)
    oob = fit.oob_score_
    if method == 'MDI':
        imp = featImpMDI(fit, featNames=trnsX.columns)
        oos = cvScore(clf, X=trnsX, y=cont['bin'], cv=cv, sample_weight=cont['w'],
                      t1=cont['t1'], pctEmbargo=pctEmbargo, scoring=scoring).mean()
    elif method == 'MDA':
        imp, oos = featImpMDA(clf, X=trnsX, y=cont['bin'], cv=cv, sample_weight=cont['w'],
                              t1=cont['t1'], pctEmbargo=pctEmbargo, scoring=scoring)
    elif method == 'SFI':
        cvGen = PurgedKFold(n_splits=cv, t1=cont['t1'], pctEmbargo=pctEmbargo)
        oos = cvScore(clf, X=trnsX, y=cont['bin'], sample_weight=cont['w'], scoring=scoring,
                      cvGen=cvGen).mean()
        clf.n_jobs = 1  # paralellize auxFeatImpSFI rather than clf
        imp = mpPandasObj(auxFeatImpSFI, ('featNames', trnsX.columns), numThreads,
                          clf=clf, trnsX=trnsX, cont=cont, scoring=scoring, cvGen=cvGen)
    return imp, oob, oos


# 8.9
def testFunc(n_features=40, n_informative=10, n_redundant=10, n_estimators=1000,
             n_samples=10000, cv=10, clf=None):
    # test the performance of the feat importance functions on artificial data
    # Nr noise features = n_features—n_informative—n_redundant
    trnsX, cont = getTestData(
        n_features, n_informative, n_redundant, n_samples)
    dict0 = {'minWLeaf': [0.], 'scoring': ['accuracy'], 'method': ['MDI', 'MDA', 'SFI'],
             'max_samples': [1.]}
    jobs, out = (dict(zip(dict0, i)) for i in product(*dict0.values())), []
    kargs = {'pathOut': './testFunc/', 'n_estimators': n_estimators,
             'tag': 'testFunc', 'cv': cv}
    for job in jobs:
        job['simNum'] = job['method']+'_'+job['scoring']+'_'+'%.2f' % job['minWLeaf'] + \
            '_'+str(job['max_samples'])
        print(job['simNum'])
        kargs.update(job)
        imp, oob, oos = featImportance(
            trnsX=trnsX, cont=cont, clf=clf, **kargs)
        plotFeatImportance(imp=imp, oob=oob, oos=oos, **kargs)
        df0 = imp[['mean']]/imp['mean'].abs().sum()
        df0['type'] = [i[0] for i in df0.index]
        df0 = df0.groupby('type')['mean'].sum().to_dict()
        df0.update({'oob': oob, 'oos': oos})
        df0.update(job)
        out.append(df0)
    out = pd.DataFrame(out).sort_values(
        ['method', 'scoring', 'minWLeaf', 'max_samples'])
    out = out['method', 'scoring', 'minWLeaf',
              'max_samples', 'I', 'R', 'N', 'oob', 'oos']
    out.to_csv(kargs['pathOut']+'stats.csv')
    return


# 8.10
def plotFeatImportance(pathOut, imp, oob, oos, method, tag=0, simNum=0, **kargs):
    # plot mean imp bars with std
    plt.figure(figsize=(10, imp.shape[0]/5.))
    imp = imp.sort_values('mean', ascending=True)
    ax = imp['mean'].plot(kind='barh', color='b', alpha=.25, xerr=imp['std'],
                          error_kw={'ecolor': 'r'})
    if method == 'MDI':
        plt.xlim([0, imp.sum(axis=1).max()])
        plt.axvline(1./imp.shape[0], linewidth=1,
                    color='r', linestyle='dotted')
    ax.get_yaxis().set_visible(False)
    for i, j in zip(ax.patches, imp.index):
        ax.text(i.get_width()/2,
                i.get_y()+i.get_height()/2, j, ha='center', va='center',
                color='black')
    plt.title('tag='+tag+' | simNum=' + str(simNum) + ' | oob=' + str(round(oob, 4)) +
              ' | oos=' + str(round(oos, 4)))
    plt.show()
    # plt.savefig(pathOut+'featImportance_'+str(simNum)+'.png', dpi=100)
    # plt.clf()
    # plt.close()
    return


# 9.2
class MyPipeline(Pipeline):
    def fit(self, X, y, sample_weight=None, **fit_params):
        if sample_weight is not None:
            fit_params[self.steps[-1][0]+'__sample_weight'] = sample_weight
        # return super(MyPipeline, self).fit(X, y, **fit_params)
        return super().fit(X, y, **fit_params)


# 9.3
def clfHyperFit(feat, lbl, t1, pipe_clf, param_grid, cv=3, bagging=[0, None, 1.],
                rndSearchIter=0, n_jobs=-1, pctEmbargo=0, scoring=None, **fit_params):
    if scoring is None:
        if set(lbl.values) == {0, 1}:
            scoring = 'f1'  # f1 for meta-labeling
        else:
            scoring = 'neg_log_loss'  # symmetric towards all cases
    # 1) hyperparameter search, on train data
    inner_cv = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)  # purged
    if rndSearchIter == 0:
        gs = GridSearchCV(estimator=pipe_clf, param_grid=param_grid,
                          scoring=scoring, cv=inner_cv, n_jobs=n_jobs, iid=False)
    else:
        gs = RandomizedSearchCV(estimator=pipe_clf, param_distributions=param_grid, scoring=scoring, cv=inner_cv, n_jobs=n_jobs,
                                iid=False, n_iter=rndSearchIter)
    gs = gs.fit(feat, lbl, **fit_params).best_estimator_  # pipeline
    # 2) fit validated model on the entirety of the data
    # mlevy edit: changing bagging[1] to bagging[0], since can't comapre None and int
    if bagging[0] > 0:
        gs = BaggingClassifier(base_estimator=MyPipeline(gs.steps),
                               n_estimators=int(bagging[0]), max_samples=float(bagging[1]),
                               max_features=float(bagging[2]), n_jobs=n_jobs)
        gs = gs.fit(feat, lbl, sample_weight=fit_params
                    [gs.base_estimator.steps[-1][0]+'__sample_weight'])
        gs = Pipeline([('bag', gs)])
    return gs


# 9.4
class logUniform_gen(rv_continuous):
    # random numbers log-uniformly distributed between 1 and e
    def _cdf(self, x):
        return np.log(x/self.a)/np.log(self.b/self.a)


# 9.4 cont.
def logUniform(a=1, b=np.exp(1)):
    ''' Example Usage:
    a,b,size=1E-3,1E3,10000
    vals=logUniform(a=a,b=b).rvs(size=size)
    print(kstest(rvs=np.log(vals),cdf='uniform',args=(np.log(a),np.log(b/a)),N=size))
    print(pd.Series(vals).describe())
    plt.subplot(121)
    pd.Series(np.log(vals)).hist()
    plt.subplot(122)
    pd.Series(vals).hist()
    plt.show()
    '''
    return logUniform_gen(a=a, b=b, name='logUniform')


# 10.1
def getSignal(events, stepSize, prob, pred, numClasses, numThreads, **kargs):
    # get signals from predictions
    if prob.shape[0] == 0:
        return pd.Series()
    # 1) generate signals from multinomial classification (one-vs-rest, OvR)
    signal0 = (prob-1./numClasses)/(prob*(1.-prob))**.5  # t-value of OvR
    signal0 = pred*(2*stats.norm.cdf(signal0)-1)  # signal=side*size
    if 'side' in events:
        signal0 *= events.loc[signal0.index, 'side']  # meta-labeling
    # 2) compute average signal among those concurrently open
    df0 = signal0.to_frame('signal').join(events[['t1']], how='left')
    df0 = avgActiveSignals(df0, numThreads)
    signal1 = discreteSignal(signal0=df0, stepSize=stepSize)
    return signal1


# 10.2
def avgActiveSignals(signals, numThreads):
    # compute the average signal among those active
    # 1) time points where signals change (either one starts or one ends)
    tPnts = set(signals['t1'].dropna().values)
    tPnts = tPnts.union(signals.index.values)
    tPnts = list(tPnts)
    tPnts.sort()
    out = mpPandasObj(mpAvgActiveSignals, ('molecule', tPnts),
                      numThreads, signals=signals)
    return out


# 10.2 cont.
def mpAvgActiveSignals(signals, molecule):
    '''
    At time loc, average signal among those still active.
    Signal is active if:
    a) issued before or at loc AND
    b) loc before signal's endtime, or endtime is still unknown (NaT).
    '''
    out = pd.Series()
    for loc in molecule:
        df0 = (signals.index.values <= loc) & (
            (loc < signals['t1']) | pd.isnull(signals['t1']))
        act = signals[df0].index
        if len(act) > 0:
            out[loc] = signals.loc[act, 'signal'].mean()
        else:
            out[loc] = 0  # no signals active at this time
    return out


# 10.3
def discreteSignal(signal0, stepSize):
    # discretize signal
    signal1 = (signal0/stepSize).round()*stepSize  # discretize
    signal1[signal1 > 1] = 1  # cap
    signal1[signal1 < -1] = -1  # floor
    return signal1


# 10.4
def betSize(w, x):
    return x*(w+x**2)**-.5


# 10.4 cont.
def getTPos(w, f, mP, maxPos):
    return int(betSize(w, f-mP)*maxPos)


# 10.4 cont.
def invPrice(f, w, m):
    return f-m*(w/(1-m**2))**.5


# 10.4 cont.
def limitPrice(tPos, pos, f, w, maxPos):
    sgn = (1 if tPos >= pos else -1)
    lP = 0
    for j in range(abs(pos+sgn), abs(tPos+1)):
        lP += invPrice(f, w, j/float(maxPos))
    lP /= tPos-pos
    return lP


# 10.4 cont.
def getW(x, m):
    # 0<alpha<1
    return x**2*(m**-2-1)


# 20.5
def linParts(numAtoms, numThreads):
    # partition of atoms with a single loop
    parts = np.linspace(0, numAtoms, min(numThreads, numAtoms)+1)
    parts = np.ceil(parts).astype(int)
    return parts


# 20.6
def nestedParts(numAtoms, numThreads, upperTriang=False):
    # partition of atoms with an inner loop
    parts, numThreads_ = [0], min(numThreads, numAtoms)
    for num in range(numThreads_):
        part = 1+4*(parts[-1]**2+parts[-1]+numAtoms*(numAtoms+1.)/numThreads_)
        part = (-1+part**.5)/2.
        parts.append(part)
    parts = np.round(parts).astype(int)
    if upperTriang:  # the first rows are heaviest
        parts = np.cumsum(np.diff(parts)[::-1])
        parts = np.append(np.array([0]), parts)
    return parts


# 20.7
def mpPandasObj(func, pdObj, numThreads=24, mpBatches=1, linMols=True, **kargs):
    '''
    Parallelize jobs, return a DataFrame or Series
    + func: function to be parallelized. Returns a DataFrame
    + pdObj[0]: Name of argument used to pass the molecule
    + pdObj[1]: List of atoms that will be grouped into molecules
    + kargs: any other argument needed by func

    Example: df1=mpPandasObj(func,('molecule',df0.index),24,**kargs)
    '''
    # not sure what this if/else does
    # or what argList is?
#     if linMols:
#         parts = linParts(len(argList[1]), numThreads*mpBatches)
#     else:
#         parts = nestedParts(len(argList[1]), numThreads*mpBatches)
    if linMols:
        parts = linParts(len(pdObj[1]), numThreads*mpBatches)
    else:
        parts = nestedParts(len(pdObj[1]), numThreads*mpBatches)

    jobs = []
    for i in range(1, len(parts)):
        job = {pdObj[0]: pdObj[1][parts[i-1]:parts[i]], 'func': func}
        job.update(kargs)
        jobs.append(job)
    if numThreads == 1:
        out = processJobs_(jobs)
    else:
        out = processJobs(jobs, numThreads=numThreads)
    if isinstance(out[0], pd.DataFrame):
        df0 = pd.DataFrame()
    elif isinstance(out[0], pd.Series):
        df0 = pd.Series()
    else:
        return out
    for i in out:
        df0 = df0.append(i)
    df0 = df0.sort_index()
    return df0


# 20.8
def processJobs_(jobs):
    # Run jobs sequentially, for debugging
    out = []
    for job in jobs:
        out_ = expandCall(job)
        out.append(out_)
    return out


# 20.9
def reportProgress(jobNum, numJobs, time0, task):
    # Report progress as asynch jobs are completed
    msg = [float(jobNum)/numJobs, (time.time()-time0)/60.]
    msg.append(msg[1]*(1/msg[0]-1))
    timeStamp = str(dt.datetime.fromtimestamp(time.time()))
    msg = timeStamp+' '+str(round(msg[0]*100, 2))+'% '+task+' done after ' + \
        str(round(msg[1], 2))+' minutes. Remaining ' + \
        str(round(msg[2], 2))+' minutes.'
    if jobNum < numJobs:
        sys.stderr.write(msg+'\r')
    else:
        sys.stderr.write(msg+'\n')
    return


def processJobs(jobs, task=None, numThreads=24):
    # Run in parallel.
    # jobs must contain a 'func' callback, for expandCall
    if task is None:
        task = jobs[0]['func'].__name__
    pool = mp.Pool(processes=numThreads)
    outputs, out, time0 = pool.imap_unordered(
        expandCall, jobs), [], time.time()
    # Process asyn output, report progress
    for i, out_ in enumerate(outputs, 1):
        out.append(out_)
        reportProgress(i, len(jobs), time0, task)
    pool.close()
    pool.join()  # this is needed to prevent memory leaks
    return out


# 20.10
def expandCall(kargs):
    # Expand the arguments of a callback function, kargs['func']
    func = kargs['func']
    del kargs['func']
    out = func(**kargs)
    return out
