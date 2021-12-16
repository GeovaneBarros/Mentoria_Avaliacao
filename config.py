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

MODEL_FINAL = Models.MLP_2_layer.value
MODEL_PARAMS = {'hidden_layer_sizes':(178, 270), 'max_iter':1000}

METHOD_FINAL = CountVectorizer()
METHOD_PARAMS = {'max_df':0.5}


Y_ENCODER = {
    'Toxic':1,
    'Non-Toxic':0
}

X_CHANGE_WORDS = {
    "'re" : ' are ',
    "'is": ' is ',
    "'ve" : ' have ',
    "n't":' not ',
    "'m": ' am ',
    "'ll": ' will '
}