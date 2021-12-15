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


