from models import Models
from sklearn.metrics import accuracy_score, f1_score, precision_score 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Define a dict of models to training
# This case i get all models right
MODELS = {elem.name:elem.value for elem in Models}

# Define a list of metrics for analysis in classifier
# This case i get all models right
METRICS = [accuracy_score, f1_score, precision_score]

# Define a list of methods of vectorizer
# This case i get all models right
METHODS = [TfidfVectorizer(), CountVectorizer()]

DF_TRAIN = './SRC/train_binary_small.csv'
DF_TEST = './SRC/test_binary_small.csv'
DF_PREDICT = './SRC/test.csv'

# Esta constante é definida para manter a padronização dos dados de y, transformando dados categóricos em numéricos
Y_CHANGE_WORDS = {
    'Toxic':1,
    'Non-Toxic':0
}

# This constant is defined after data analisys. Irá ser utilizada para o preprocessamento 
X_CHANGE_WORDS = {
    "'re" : ' are ',
    "'is": ' is ',
    "'ve" : ' have ',
    "n't":' not ',
    "'m": ' am ',
    "'ll": ' will '
}