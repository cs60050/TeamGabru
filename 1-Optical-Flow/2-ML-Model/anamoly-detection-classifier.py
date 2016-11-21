from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

from sklearn import metrics

import os
import codecs
import numpy as np
import pickle

basepath = os.path.dirname(os.path.abspath(__file__))+"/../featueExtraction/Output"
model_path = os.path.dirname(os.path.abspath(__file__))+"/TrainedClassifiers"
output_path = os.path.dirname(os.path.abspath(__file__))+"/Output"
eval_path = os.path.dirname(os.path.abspath(__file__))+"/Evaluation"


names = ["DecisionTree"]





evaluation_names = ["Accuracy","F1 Score","F1_Micro","F1_Macro","F1_Weighted","Log_Loss","Precision","Recall","ROC_AUC"]

def evaluate(y_true,y_pred):
	return [accuracy_score(y_true, y_pred),
	f1_score(y_true, y_pred, average=None),
	f1_score(y_true, y_pred, average='micro'),
	f1_score(y_true, y_pred, average='macro'),
	f1_score(y_true, y_pred, average='weighted'),
	log_loss(y_true,y_pred),
	precision_score(y_true, y_pred, average=None),
	recall_score(y_true, y_pred, average=None),
	roc_auc_score(y_true, y_pred)]


def auc_and_eer(y_true, y_pred):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    return [metrics.auc(fpr, tpr), EER(fpr, tpr)]



def EER(fpr, tpr):
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer


def load_train_dataset(train_path):
    files = os.listdir(train_path)

    X_train = []
    y_train = []

    for filename in files:
        if filename == ".DS_Store":
            continue
        file = codecs.open(train_path+"/"+filename,'r','utf-8')

        for row in file:
            l = row.strip().split(",")
            X_train.append(l[0:11])
            y_train.append(int(l[11]))
        print(filename)

    return X_train,y_train


def load_test_dataset(test_path):
    files = os.listdir(test_path)

    X_test = []
    y_true = []

    for filename in files:
        if filename == ".DS_Store":
            continue
        file = codecs.open(test_path+"/"+filename,'r','utf-8')

        for row in file:
            l = row.strip().split(",")
            X_test.append(l[0:11])
            y_true.append(int(l[11]))
        print(filename)

    return X_test,y_true


def plot():
     #Two subplots, unpack the axes array immediately
    f, ax1= plt.subplots(1)

    ax1.scatter(X_vis[y == 0, 0], X_vis[y == 0, 1], label="Class #0", alpha=0.5,
                edgecolor=almost_black, facecolor=palette[0], linewidth=0.15)
    ax1.scatter(X_vis[y == 1, 0], X_vis[y == 1, 1], label="Class #1", alpha=0.5,
                edgecolor=almost_black, facecolor=palette[2], linewidth=0.15)
    ax1.set_title('Original set')


def main():

    treshold_dirs = os.listdir(basepath)

    for dir in treshold_dirs:
        if dir == ".DS_Store":
            continue
        print(dir)
        ped_dirs = os.listdir(basepath+"/"+dir)

        for sub_dir in ped_dirs:
            if sub_dir == ".DS_Store":
                continue

            print(dir,sub_dir)

            train_path = basepath+"/"+dir+"/"+sub_dir+"/Train"
            test_path = basepath+"/"+dir+"/"+sub_dir+"/Test"

            write_file = codecs.open(output_path+"/"+dir+"_"+sub_dir+"-output.txt",'w','utf-8')
            eval_file = codecs.open(eval_path+"/"+dir+"_"+sub_dir+"-evaluation_scores.txt",'w','utf-8')

            X_train,y_train = load_train_dataset(train_path)

            X_test,y_true = load_test_dataset(test_path)

            # pca = PCA(n_components = 2)
            # X_red, y_red = pca.fit_transform(X_train, y_train)



            print(train_path,test_path)
            classifiers = [
                DecisionTreeClassifier(max_depth=5)]

            for algo, clf in zip(names, classifiers):
                try:
                    with open(model_path+"/"+dir+"/"+sub_dir+"/"+algo + '.pkl', 'rb') as f1:
                        clf = pickle.load(f1)
                except:
                    clf.fit(X_train, y_train)
                    with open(model_path+"/"+dir+"/"+sub_dir+"/"+algo + '.pkl', 'wb') as f1:
                        pickle.dump(clf, f1)

                predicted = []
                print(algo+"_fitted")

                for ind in range(0,len(X_test)):
                    try:
                        vector = np.matrix(X_test[ind])
                        predicted+=[clf.predict(vector)[0]]

                    except:
                        print("Error")

                print(algo, predicted, file=write_file)
                print(algo+"_Tested")

                report = metrics.classification_report(y_true, predicted)
                print(algo,file=eval_file)
                print(report, file = eval_file)

                #print(evaluate(y_test, y_hat))
                print(auc_and_eer(y_true, predicted), file = eval_file)
                

                # scores = evaluate(y_true,predicted)
                # print(algo+"\t"+str(scores),file=eval_file)


if __name__ == "__main__":main()
