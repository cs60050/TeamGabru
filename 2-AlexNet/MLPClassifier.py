#from sklearn.neural_network import MLPClassifier
import ast
import numpy as np
import scipy
import pickle
import os

from collections import Counter

from time import time
from collections import defaultdict
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import plotly

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc


#from sklearn.pipeline import Pipeline

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel


from imblearn.under_sampling import (EditedNearestNeighbours, RandomUnderSampler,
                                     RepeatedEditedNearestNeighbours)
from imblearn.ensemble import EasyEnsemble
from imblearn.pipeline import Pipeline as im_Pipeline

import rank_scorers
import sampler
import feature_importance
import useClaimBuster
import dataset_utils


basepath = "/home/bt1/13CS10060/btp"
datapath = basepath+"/ayush_dataset"
workingdir = basepath + "/output_all"

import seaborn as sns
sns.set()

# Define some color for the plotting
almost_black = '#262626'
palette = sns.color_palette()

names = [
"KNN", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis",
         "Quadratic Discriminant Analysis", 
         "MLP"]


classifiers = [
    KNeighborsClassifier(weights='distance', n_neighbors=121),
    SVC(kernel="linear", C=1, probability=True),
    SVC(C=1, probability=True),
    DecisionTreeClassifier(max_depth=10),
    RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(solver='lsqr', shrinkage="auto"),
    QuadraticDiscriminantAnalysis(),
    MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100,75,50,25,15), max_iter=10000, random_state=1)
    ]


param_grid = {
    "Linear SVM" : {
        'C': [1,5,10,100,500,1000],
    },
    "RBF SVM" : {
        'C': [1,5,10,100,500,1000],
        'gamma': [1e-5, 1e-4,1e-3,1e-2,1e-1],
        'kernel': ['poly', 'sigmoid'],
        'degree': [3,5,8,10]
    },
    "KNN" : {
        'weights': ['distance'],
        'n_neighbors': [1,10,50,100]
    }
}

'''
names = ["Decision Trees", "Neural Networks"]


classifiers = [
    DecisionTreeClassifier(),
    MLPClassifier(algorithm='adam', alpha=1e-5, hidden_layer_sizes=(15, 15), random_state=1, verbose=True)
    ]'''

evaluation_names = ["Accuracy","F1 Score","F1_Micro","F1_Macro","F1_Weighted","Log_Loss","Precision","Recall","ROC_AUC"]

evaluation_methods = []

def evaluate(y_true,y_pred):
	return [accuracy_score(y_true, y_pred),
	f1_score(y_true, y_pred, average="binary"),
	#f1_score(y_true, y_pred, average='micro'),
	#f1_score(y_true, y_pred, average='macro'),
	#f1_score(y_true, y_pred, average='weighted'),
	#log_loss(y_true,y_pred),
	precision_score(y_true, y_pred, average="binary"),
	recall_score(y_true, y_pred, average="binary"),
	roc_auc_score(y_true, y_pred)]



def load_dataset(trainfilelist, indexlist):
    x = []
    Y = []
    allfeatures = []
    embed_feats = []
    allindex = []
    names = []
    for i,files in enumerate(trainfilelist):
        f1 = open(files[0], "r")
        f3 = open(files[1], 'r')
        f2 = open(indexlist[i], "r")
        names = f1.readline()
        names = names.strip().split(" ")[:-1]
        # names = names[:60]
        for lines in f1:
            features = [float(value) for value in lines.split(' ')]
            # features = features[:60] + [features[-1]]
            # print(features)
            allfeatures.append(features)
        for lines in f2:
            indexes = [int(value) for value in lines.split(' ')]
            allindex.append(indexes)
        for lines in f3:
            embeds = [float(value) for value in lines.split(" ")]
            embed_feats.append(embeds)


    # from random import shuffle
    # shuffle(allfeatures)
    n = ["embed"+str(i) for i in range(300)]
    n.extend(names)
    print(len(allfeatures[0]))

    for embeds,feature in zip(embed_feats, allfeatures):
        f = []
        f.extend(embeds)
        f.extend(feature[:-1])
        x.append(f)
        #print(feature[-1])
        Y.append(feature[-1])
    # print(len(names),len(feature))
    # print(Y.count(1))
    # exit(0)
    return n,x, Y, allindex


def feature_select(X,y):
    clf = ExtraTreesClassifier()
    clf = clf.fit(X, y)
    print(clf.feature_importances_)
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    print(X_new.shape) 
    return X_new, y


def plot_data_and_sample(X,y, sampler):
    # Instanciate a PCA object for the sake of easy visualisation
    # pca = PCA(n_components=2)

    # # Fit and transform x to visualise inside a 2D feature space
    # X_vis = pca.fit_transform(X)

    # X_resampled, y_resampled = sampler.fit_sample(X, y)

    # print(len(X_resampled), len(y_resampled))
    X_vis = X

    X_res_vis = [X]
    y_resampled=y
    # for X_res in X_resampled:
    #     X_res_vis.append(pca.transform(X_res))

    # Two subplots, unpack the axes array immediately
    # f, (ax1, ax2) = plt.subplots(1, 2)

    # ax1.scatter(X_vis[y == 0, 0], X_vis[y == 0, 1], label="Class #0", alpha=0.5,
    #             edgecolor=almost_black, facecolor=palette[0], linewidth=0.15)
    # ax1.scatter(X_vis[y == 1, 0], X_vis[y == 1, 1], label="Class #1", alpha=0.5,
    #             edgecolor=almost_black, facecolor=palette[2], linewidth=0.15)
    # ax1.set_title('Original set')

    # ax2.scatter(X_vis[y == 0, 0], X_vis[y == 0, 1], label="Class #0", alpha=0.5,
    #             edgecolor=almost_black, facecolor=palette[0], linewidth=0.15)
    # for iy, e in enumerate(X_res_vis):
    #     ax2.scatter(e[y_resampled[iy] == 1, 0], e[y_resampled[iy] == 1, 1],
    #                 label="Class #1", alpha=0.5, edgecolor=almost_black,
    #                 facecolor=np.random.rand(3,), linewidth=0.15)
    # ax2.set_title('Easy ensemble')

    # plt.show()

    print(X)
    X_vis0 = X_vis[y==0, 0]
    X_vis1 = X_vis[y==1, 0]
    X_vis0 = X_vis0.tolist()
    X_vis1 = X_vis1.tolist()
    X_vis0_probs_dict = {x:X_vis0.count(x)/len(X_vis0) for x in X_vis0}
    X_vis0_probs = X_vis0_probs_dict.values()
    X_vis1_probs_dict = {x:X_vis1.count(x)/len(X_vis1) for x in X_vis1}
    X_vis1_probs = X_vis1_probs_dict.values()
    # print(list(X_vis0_probs))
    # print(list(range(100)))
    # exit(0)
    trace1 = go.Scatter(
                x = list(range(62)),
                y=list(X_vis0_probs),
                name='Non Check-worthy'
                # histnorm='probability'
             )
    trace2 = go.Scatter(
                x = list(range(62)),
                y=list(X_vis1_probs),
                name='Check-worthy'
                # histnorm='probability'
             )
    data = [trace1, trace2]
    layout = go.Layout(
            showlegend=True,
            legend = dict(
                x = 0.6,
                y = 1
                ),
            width=450,
            height=400,
            xaxis=dict(title='Length of sentence'),
            yaxis=dict(title='Probability')
            )
    # fig = dict(data=data)
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig,image='png')
    exit(0)
    return X_resampled, y_resampled



def plot_ROC_curve(roc_curve):
    false_positive_rate, true_positive_rate, _ = roc_curve
    roc_auc = 0
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
    label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def plot_PR_curve(pr_curve):
    precision, recall, _ = pr_curve
    plt.plot(recall, precision, lw=2, color='navy',
             label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall')
    plt.legend(loc="lower left")
    plt.show()



# Utility function to report best scores
def report(results, n_top=10):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")



def cross_validate(X,y):
    for name, clf in zip(names[1:3], classifiers[1:3]):
        
        scores = cross_val_score(clf, X, y, cv=4)
        print("Name %s ROC_AUC: %0.2f (+/- %0.2f)" % (name, scores.mean(), scores.std() * 2))



def randomGridSearch(X,y):
    for name, clf in zip(names[1:3], classifiers[1:3]):
        # run randomized search
        n_iter_search = 2
        random_search = RandomizedSearchCV(clf, param_distributions=param_grid[name],
                                           n_iter=n_iter_search)

        start = time()
        random_search.fit(X, y)
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
              " parameter settings." % ((time() - start), n_iter_search))
        report(random_search.cv_results_)


def gridSearch(X,y, working_dir):
    for name, clf in zip(names[0:1], classifiers[0:1]):
        # run grid search
        clf = GridSearchCV(clf, param_grid=param_grid[name],cv=4, scoring="roc_auc" ,n_jobs=24)

        start = time()
        clf.fit(X, y)
        with open(working_dir + "/grid_best_2" + name + '.pkl', 'wb') as f1:
                pickle.dump(clf, f1)
        print("GridSearchCV took %.2f seconds candidates"
              " parameter settings." % ((time() - start)))
        report(clf.cv_results_)


def normalize_topic_values(X, y):
    X[X<1e-4] = 0
    return X,y



def split_data(X,y, index, frac=0.2):
    from collections import Counter
    from sklearn.utils import shuffle
    import random
    c = Counter()

    n = len(X)

    X=np.asarray(X)
    y = np.asarray(y)
    index = np.asarray(index)



    for i in range(n):
        if(y[i] == 1):
            c[index[i][0]] += 1;
    l = list(c.items())
    l = shuffle(l, random_state=101)

    test_debates = []

    test_size = int(frac* sum(y))

    k = 0
    while(test_size > 0):
        test_debates.append(l[k][0])
        test_size -= l[k][1]
        k +=1

    print(test_size, test_debates)

    X_test = []
    y_test = []
    X_train = []
    y_train = []
    index_test = []
    index_train = []


    for i in np.random.permutation(n):
        if(index[i][0] in test_debates):
            X_test.append(X[i])
            y_test.append(y[i])
            index_test.append(index[i])
        else:
            X_train.append(X[i])
            y_train.append(y[i])
            index_train.append(index[i])

    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    index_test = np.asarray(index_test)
    index_train = np.asarray(index_train)


    print(np.shape(X_train))


    p = np.random.permutation(len(X_train))
    test_p = np.random.permutation(len(X_test))

    return X_train[p], X_test[test_p], y_train[p], y_test[test_p], index_train[p], index_test[test_p]



def evaluate(X_test, y_test, index_test, clf, name, sent_print=True):

    y_hat = clf.predict(X_test)
    report = metrics.classification_report(y_test, y_hat)
    #print(str(score))
    f = open(working_dir + "/" + name + '_report.txt', 'w')
    f.write(name+"\n")
    f.write(report)
    print(report)

    # try:
    #     plot_ROC_curve(metrics.roc_curve(y_test, clf.decision_function(X_test), pos_label=1))
    #     plot_PR_curve(metrics.precision_recall_curve(y_test,clf.decision_function(X_test), pos_label=1 )) 
    # except:
    #     pass

    try:
        y_prob = clf.predict_proba(X_test)[:,1]
    except:
        pass

    ks = [10,20,30,40,50,60,70,80,90,100,200,300,500,1000]

    allscores = rank_scorers.all_score(y_test, y_prob, ks)

    
        
    for i,k in enumerate(ks):
        print(k,round(allscores[i][0],3),round(allscores[i][1],3),round(allscores[i][2],3), sep="\t", file=f)

    #print(allscores)

    if(not sent_print):
        return
    sent_list = [dataset_utils.get_sentence(idx) for idx in index_test]
    ff = open(working_dir + "/" + name + '_scores.txt', 'w')
    for tag, score, sent in zip(y_test, y_prob, sent_list):
        print(tag, score, sent, sep="\t", file=ff)

    # buster_prob = dataset_utils.get_buster_score(index_test)

    # allscores_buster = rank_scorers.all_score(y_test, buster_prob, ks)

    # # for tag, score, sent in zip(y_test, buster_prob, sent_list):
    # #     print(tag, score, sent, sep="\t")
    # print("ClaimBuster",file=f)
    # for i,k in enumerate(ks):
    #     print(k,round(allscores_buster[i][0],3),round(allscores_buster[i][1],3),round(allscores_buster[i][2],3), sep="\t", file=f)


def ensemble_train(X,y, working_dir,n, name, svm=True):
    ees = EasyEnsemble(random_state=557, n_subsets=n)
    X_res, y_res = ees.fit_sample(X,y)
   

    try:
        raise Exception('Retrain')
        with open(working_dir + "/" + name  + '.pkl', 'rb') as f1:
            clf = pickle.load(f1)
    except:
        # scores = cross_val_score(clf, X, y, cv=4, scoring="roc_auc")
        # print("Name %s ROC_AUC: %0.2f (+/- %0.2f)" % (name, scores.mean(), scores.std() * 2))
        clf = []
        for i in range(len(X_res)):
            print(Counter(y_res[i]))
            if(svm):
                clfi = SVC(kernel="linear", probability=True)
            else:
                clfi = AdaBoostClassifier(n_estimators=20)
            #clfi=AdaBoostClassifier()
            clfi.fit(X_res[i], y_res[i])
            clf.append(clfi)
            scores = cross_val_score(clfi, X_res[i], y_res[i], cv=4, scoring="roc_auc")
            print("Name %s ROC_AUC: %0.2f (+/- %0.2f)" % (name, scores.mean(), scores.std() * 2))
        with open(working_dir + "/" + name + '.pkl', 'wb') as f1:
            pickle.dump(clf, f1)  
    return clf

def ensemble_predict_proba(clf, X):
    y_proba = []
    for clfi in clf:
        y_probai = clfi.predict_proba(X)[:,-1]
        y_proba.append(y_probai)

    y_proba = np.asarray(y_proba)

    y_proba_mean = np.mean(y_proba, axis=0)

    y_hat = np.round(y_proba_mean)

    return y_proba_mean, y_hat




sel_classifiers = [
    SVC(kernel="linear", C=1, probability=True),
    SVC(C=1, probability=True),
    RandomForestClassifier(n_estimators=20),
    AdaBoostClassifier(n_estimators=10)
    ]

sel_names = ["lsvm", "rsvm", "rfc", "ada"]

def main(working_dir, args):
    f_names, X,y, index = load_dataset([(workingdir+"/features.ff", workingdir+"/embeddings.txt")], [workingdir+"/index.txt"])

    print(len(X), len(y))    

    
    X = np.asarray(X)
    y = np.asarray(y)
    index = np.asarray(index)
    f_names = np.asarray(f_names)
    start = 300
    X_part, y = normalize_topic_values(X[start:],y)

    X[start:] = X_part[:]

    print(np.shape(X), np.shape(f_names))
    print(X[0])
    # sel_feats = np.asarray(list(range(0,300)))# + list(range(413,414)))
    #sel_feats = np.asarray(list(range(300,len(X[0]))))
    sel_feats = np.asarray(list(range(0,300)))
    X_posonly = X[:,sel_feats]

    print(np.shape(X_posonly))
    f_names = f_names[sel_feats] 

    print(f_names)
    # index_no = index[y==0]

    # print(np.shape(index_no))

    # index_no_sampled = index_no[np.random.choice(range(len(index_no)), size=50, replace=False)]

    # for indexi in index_no_sampled:
    #     print(dataset_utils.get_sentence(indexi))



    # plot_data_and_sample(X_posonly,y,None)

    # feature_importance.plot_feature_importance(X_posonly,y,f_names)
    # exit(0)

    


    #exit(0)



    #feature_importance.plot_feature_importance(X,y,f_names)

    #exit(0)
      
    X_train, X_test, y_train, y_test, index_train, index_test = split_data(X_posonly, y, index)

    pca = PCA(n_components=100)
    X_train = pca.fit_transform(X_train)
    
    print(np.shape(X_train))

    X_test = pca.transform(X_test)

    X_vis= X_train


    # #Two subplots, unpack the axes array immediately
    # f, ax1 = plt.subplots(1)

    # ax1.scatter(X_vis[y_train == 0, 0], X_vis[y_train == 0, 1], label="Class #0", alpha=0.5,
    #             edgecolor=almost_black, facecolor=palette[0], linewidth=0.15)
    # ax1.scatter(X_vis[y_train == 1, 0], X_vis[y_train == 1, 1], label="Class #1", alpha=0.5,
    #             edgecolor=almost_black, facecolor=palette[2], linewidth=0.15)
    # ax1.set_title('Original set')


    # plt.show()

    # gridSearch(X,y, working_dir)
    rsampler = RandomUnderSampler(random_state=487)
    X_test_s, y_test_s = rsampler.fit_sample(X_test, y_test)
    
    #sampler= EasyEnsemble()
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=44)
    ensemble_clf = [True, False]
    for c in ensemble_clf:
        n = 20
        name = "embed50svm_"+str(n)+str(c)
        fr = open(working_dir+"/"+name+"report.txt", "w")
        clf = ensemble_train(X_train, y_train, working_dir,n, name, c)
        y_prob, y_hat = ensemble_predict_proba(clf, X_test)

        y_prob_s, y_hat_s = ensemble_predict_proba(clf, X_test_s)

        report = metrics.classification_report(y_test, y_hat)
        #print(str(score))
        print(report)
        print(report, file=fr)

        #evaluate(X_test, y_test, index_test, clf, name) 

        ks = [10,20,30,40,50,60,70,80,90,100,200,300,500,1000]

        allscores = rank_scorers.all_score(y_test, y_prob, ks)

        
        for i,k in enumerate(ks):
            #print(k,round(allscores[i][0],3),round(allscores[i][1],3),round(allscores[i][2],3), sep="\t")
            print(k,round(allscores[i][0],3),round(allscores[i][1],3),round(allscores[i][2],3), sep="\t", file=fr)

        report = metrics.classification_report(y_test_s, y_hat_s)
        #print(str(score))
        print(report)
        print(report, file=fr)

        #evaluate(X_test, y_test, index_test, clf, name) 

        ks = [10,20,30,40,50,60,70,80,90,100,200,300,500,1000]

        allscores = rank_scorers.all_score(y_test_s, y_prob_s, ks)

        
        for i,k in enumerate(ks):
            #print(k,round(allscores[i][0],3),round(allscores[i][1],3),round(allscores[i][2],3), sep="\t")
            print(k,round(allscores[i][0],3),round(allscores[i][1],3),round(allscores[i][2],3), sep="\t", file=fr)




        #print(allscores)
        
        # sent_list = [dataset_utils.get_sentence(idx) for idx in index_test]
        # f = open(working_dir+"/"+name+"scores.txt", "w")
        # for tag, score, sent in zip(y_test, y_prob, sent_list):
        #     print(tag, score, sent, sep="\t", file=f)

        #buster_prob = dataset_utils.get_buster_score(index_test)

        #allscores_buster = rank_scorers.all_score(y_test, buster_prob, ks)

        # for tag, score, sent in zip(y_test, buster_prob, sent_list):
        #     print(tag, score, sent, sep="\t")
        # print("ClaimBuster")
        # for i,k in enumerate(ks):
        #     #print(k,round(allscores_buster[i][0],3),round(allscores_buster[i][1],3),round(allscores_buster[i][2],3), sep="\t")
        #     print(k,round(allscores_buster[i][0],3),round(allscores_buster[i][1],3),round(allscores_buster[i][2],3), sep="\t", file=fr)



    rsampler = RandomUnderSampler(random_state=487)
    X_train, y_train = rsampler.fit_sample(X_train, y_train)

    #X_train, X_test = feature_importance.recursive_elimination(X_train,y_train, X_test)



    X_test_s, y_test_s = rsampler.fit_sample(X_test, y_test)

   
    #for h in [1]:
        
    for name, clf in zip(sel_names, classifiers):
        print(name)
        name="embed50"+name
        #clf = SVC(probability=True, kernel="linear")
        # pipe_components[-1] = ('classification', clf)
        # clf = im_Pipeline(pipe_components)
        #print(clf)

        try:
            raise Exception('Retrain')
            with open(working_dir + "/" + name + '.pkl', 'rb') as f1:
                clf = pickle.load(f1)
        except:
            scores = cross_val_score(clf, X_train, y_train, cv=4, scoring="roc_auc")
            #rec_scores = cross_val_score(clf, X_train, y_train, cv=4, scoring="roc_auc")
            print("Name %s ROC_AUC: %0.2f (+/- %0.2f)" % (name, scores.mean(), scores.std() * 2))
            clf.fit(X_train, y_train)
            with open(working_dir + "/" + name + '.pkl', 'wb') as f1:
                pickle.dump(clf, f1)  
        evaluate(X_test, y_test, index_test, clf, name)
        evaluate(X_test_s, y_test_s, index_test, clf, name+"sampledtest", False )        
    


if __name__ == '__main__':
    import os
    import sys


    working_dir = workingdir+"/models_feat/finals_2" #os.argv[-1]
    try:
        os.makedirs(working_dir)
    except:
        pass

    arguments = sys.argv[1:]
    args = defaultdict(None)
    for x in arguments:
        x = x.split("=")
        args[x[0].strip("-")] = x[1]

    main(working_dir, args)
