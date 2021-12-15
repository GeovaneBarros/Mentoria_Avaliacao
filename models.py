from enum import Enum

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

class Models(Enum):
    KNN = KNeighborsClassifier()
    SVC_LINEAR =  SVC(kernel='linear')
    SVC_RBF =  SVC()
    DECISION_TREE =  DecisionTreeClassifier()
    MLP_1_layer = MLPClassifier(max_iter=500)
    MLP_2_layer = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=500)
    RANDOM_FOREST = RandomForestClassifier()
    XGBOOST = XGBClassifier(use_label_encoder=False,eval_metric = "logloss", learning_rate = 0.1)
    GRADIENT_BOOST =  GradientBoostingClassifier()
    BAYES_MULTINOMIAL =  MultinomialNB()
    EXTRA_TREE = ExtraTreesClassifier()
    LOGISTIC_REGRESSION = LogisticRegression()