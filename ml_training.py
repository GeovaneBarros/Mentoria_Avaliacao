import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

def simple_training(X_train, y_train, X_test, y_test, models, list_metrics):
    df = {
        'Model': [],
    }
    for metric in list_metrics:
        name = metric.__name__
        df[name] = []

    for model in models:
        model_training = models[model].fit(X_train, y_train)
        y_pred = model_training.predict(X_test)
        df['Model'].append(model)
        for metric in list_metrics:
            name = metric.__name__
            df[name].append(metric(y_test,y_pred))
            
    return pd.DataFrame(df)

def full_training(X_train, y_train, X_test, y_test, models, params, list_metrics, n_iter=5):
    
    df = {
        'Model': [],
        'Params': [],
    }
    for metric in list_metrics:
        name = metric.__name__
        df[name] = []

    for model in models:
        model_training = RandomizedSearchCV(models[model], params[model], n_iter=n_iter, random_state=10, cv=3).fit(X_train, y_train)
        y_pred = model_training.predict(X_test)
        df['Model'].append(model)
        df['Params'].append(model_training.best_params_)

        for metric in list_metrics:
            name = metric.__name__
            df[name].append(metric(y_test,y_pred))
            
    return pd.DataFrame(df)


def final_training(X_train, y_train, X_test, y_test, model, params, list_metrics):
    df = {}
    for metric in list_metrics:
        name = metric.__name__
        df[name] = ''

    model_training = GridSearchCV(model, params, cv=2).fit(X_train, y_train)
    y_pred = model_training.predict(X_test)
    
    df['Params'] = model_training.best_estimator_
    df['Estimator'] = model_training.best_estimator_
    
    for metric in list_metrics:
                name = metric.__name__
                df[name] = metric(y_test,y_pred)

    return df