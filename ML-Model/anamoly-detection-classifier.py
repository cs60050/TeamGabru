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


names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis",
         "Quadratic Discriminant Analysis"]


classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]


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
	roc_auc_score(y_true, y_scores)]



def load_train_dataset():


def load_test_dataset():


def main():

	X_train,y_train = load_train_dataset()

	X_test,y_test = load_test_dataset()

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        
        clf.fit(X_train, y_train)

        score = clf.score(X_test, y_test)


if __name__ == "__main__":main()