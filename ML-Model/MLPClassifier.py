from sklearn.neural_network import MLPClassifier
import ast
import numpy as np
import pickle
import os

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

Train1 = 'Features/10/Ped1/Train/'
Train2 = 'Features/10/Ped2/Train/'
Train3 = 'Features/50/Ped1/Train/'
Train4 = 'Features/50/Ped2/Train/'

Test1 = 'Features/10/Ped1/Test/'
Test2 = 'Features/10/Ped2/Test/'
Test3 = 'Features/50/Ped1/Test/'
Test4 = 'Features/50/Ped2/Test/'

names = ["Decision Trees", "Neural Networks"]


classifiers = [
    DecisionTreeClassifier(),
    MLPClassifier(algorithm='adam', alpha=1e-5, hidden_layer_sizes=(15, 15), random_state=1, verbose=True)
    ]

evaluation_names = ["Accuracy","Avg. Precision","F1 Score","F1_Micro","F1_Macro","F1_Weighted","Log_Loss","Precision","Recall","ROC_AUC"]

evaluation_methods = []

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



def load_train_dataset():
    x = []
    Y = []
    for files in os.listdir(Train1):
        f = open(Train1 + files, 'r')
        for lines in f:
            lines = lines.split(',')
            i = 0
            x1 = []
            while i<11:
                x1.append(float(lines[i]))
                i += 1
            label = int(lines[11])
            x.append(x1)
            Y.append(label)
    for files in os.listdir(Train2):
        f = open(Train2 + files, 'r')
        for lines in f:
            lines = lines.split(',')
            i = 0
            x1 = []
            while i<11:
                x1.append(float(lines[i]))
                i += 1
            label = int(lines[11])
            x.append(x1)
            Y.append(label)

    return x, Y


def load_test_dataset():
    x = []
    Y = []
    for files in os.listdir(Test1):
        f = open(Test1 + files, 'r')
        for lines in f:
            lines = lines.split(',')
            i = 0
            x1 = []
            while i<11:
                x1.append(float(lines[i]))
                i += 1
            label = int(lines[11])
            x.append(x1)
            Y.append(label)
    # for files in os.listdir(Test2):
    #     f = open(Test2 + files, 'r')
    #     for lines in f:
    #         lines = lines.split(',')
    #         i = 0
    #         x1 = []
    #         while i<11:
    #             x1.append(float(lines[i]))
    #             i += 1
    #         label = int(lines[11])
    #         x.append(x1)
    #         Y.append(label)

    return x, Y


def main():
    X_train,y_train = load_train_dataset()
    X_test,y_test = load_test_dataset()
    for name, clf in zip(names, classifiers):
        print(name)
        coll = []
        try:
            with open(name + '.pkl', 'rb') as f1:
                clf = pickle.load(f1)
        except:
            clf.fit(X_train, y_train)
            with open(name + '.pkl', 'wb') as f1:
                pickle.dump(clf, f1)
        for vals in X_test:
            z = clf.predict([vals])
            coll.append(z[0])
        score = evaluate(y_test, coll)
        print(str(score))
        f = open(name + '.txt', 'a')
        f.write(str(score))


if __name__ == '__main__':
    main()
