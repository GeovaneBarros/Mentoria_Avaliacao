import ml_training
from sklearn.pipeline import Pipeline

def pipeline_models(dict_models, method):
    models = {}
    for key in dict_models:
        models[key] = Pipeline([
            ('vectorizer', method),
            ('clf', dict_models[key])
        ])
    return models

def simple_training(X_train, y_train, X_test, y_test,dict_models, list_metrics, method):
    models = pipeline_models(dict_models, method)
    return  ml_training.simple_training(X_train, y_train,X_test,y_test, models, list_metrics)   
    
def full_training(X_train, y_train, X_test, y_test, dict_models, params , list_metrics, method, n_iter=5):
    models = pipeline_models(dict_models, method)
    return ml_training.full_training(X_train, y_train,X_test,y_test, models, params, list_metrics, n_iter=n_iter)

def final_training(X_train, y_train, X_test, y_test, model, params, metric, method):
    model = Pipeline([
        ('vectorizer', method),
        ('clf', model)
    ])
    return ml_training.final_training(X_train, y_train, X_test, y_test, model, params, metric)