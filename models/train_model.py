import pandas as pd
import numpy as np
import os
import sys
import shap
import json
import mlflow
import mlflow.sklearn
from datetime import datetime
from sklearn.model_selection import train_test_split,cross_val_score, StratifiedKFold
# we'll use StandardScaler to put the features on the same scale
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay)
from mapie.classification import _MapieClassifier as MapieClassifier 
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_feature_matrix():

    csv_path = os.path.join(
        os.path.dirname(__file__),
        '..',       # the parent folder
        'features/data',     # the folder inside the parent folder
        'feature_matrix.csv'    # the file we are looking at
    )

    df = pd.read_csv(csv_path)
    print(f"We have {df.shape[1]} columns and {df.shape[0]} rows")
    # spliting the features and the target
    X = df.drop('converted', axis=1)
    y = df['converted']

    os.makedirs('models', exist_ok=True)
    with open('models/feature_names.json', 'w') as f:
        json.dump(list(X.columns), f)
    print(f"Feature names saved to models/feature_names.json")
    
    return X, y

def split_and_scale(X, y):
    """
    Split the data into train/test categories and
    putting the features into one scale.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size = 0.2,
        random_state = 42,
        # keeping class distribution in both categories(train/test)
        stratify = y
    )

    print(f"Train set: {X_train.shape[0]} leads.")
    print(f"Test set: {X_test.shape[0]} leads.")

    # After spliting we calculate mean,std of train data
    # Then we scale the train and test data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs('models', exist_ok=True)
    
    # we used jobli.dump() to save some object with large numerical data in a file
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Scaler saved to models/scaler.pkl")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def get_models():
    """
    Returns a dictionary of the 3 models we want to train and compare.

    - we use a dictionary so we can loop through all models and to avoid the repetition.
    - Each model has class_weight='balanced'.
    - we use class_weight='balanced' to tell the model to pay attention for the rare cases.
    """

    models = {
        # Here we use the logistic Regression to separate the converted ones from the non-converted.
        'Logistic Regression': LogisticRegression(
            max_iter=1000, # This para to use enough iterations to find the best line.
            class_weight='balanced', # this one to mention that we care for the rare cases.
            random_state=42
        ),

        # Here we use the Decision trees to try to use many trees on random features. 
        'Random Forest': RandomForestClassifier(
            n_estimators=200, # this one to get 200 tree.
            class_weight='balanced',
            random_state=42,
            n_jobs=-1   # to use all the resources to train faster.
        ),

        # Here we use XGBoost which based on Gradient Boosting.
        # It is used to create trees sequentially so it learns from each previous one.
        # It's formula: num of negatives / num of positives
        'XGBoost': XGBClassifier(
            n_estimators=200,
            scale_pos_weight=2.7, # the same function as 'balanced' and in our case we got num.n/num.p=2.7
            random_state=42,
            eval_metric='logloss', # we use logloss to measure how much wrong the tree from the previous.
            verbosity=0 # we use 0 so it don't print the result of each tree
        )
    }

    return models

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Trains and evalute all the 3 models.
    It returns:
    - results dict: metrics for each model.
    - trained_models dict: the actual trained model objects.
    """

    models = get_models()
    results = {}
    trained_models = {}

    print("Training & Evaluating all models!")
    print('#'*30)

    for model_name, model in models.items():
        print(f"\n{model_name}")
        print('#'*30)
        print('Training...')
        model.fit(X_train, y_train) # finds the patterns
        y_pred = model.predict(X_test) # returns predictions {0, 1}
        # predict_proba() returns probabilities like [0.15, 0.85]
        # we take just the probabilities of (classe 1)
        y_prob = model.predict_proba(X_test)[:, 1]

        # --- Metrics calculation ---
        # ROC-AUC measures how well the model separates the two classes
        # the more result we got more the model is perfect.
        auc = roc_auc_score(y_test, y_prob)

        # classification_report returns precision, recall, f1 as a dict
        # we use output_dict so we can make the result readable for the machine.
        report = classification_report(y_test, y_pred, output_dict=True)

        # Extract class 1 metric
        # We use precision to see from all the leads we predicted to convert, how many did?
        precision = report['1']['precision']
        # High precision = Few false alarms

        # we use Recall to see from all leads that actually converted. how many did we identify?
        recall = report['1']['recall']
        # High recall = we don't miss valuable leads

        # we use the F1-score to measure the mean of precision and recall to balance them
        f1 = report['1']['f1-score']

        # To store all the metrics for each model
        results[model_name] = {
            'roc_auc': round(auc, 4), # give 4 numbers after the comma.
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'model': model
        }

        trained_models[model_name] = model

        print(f"    ROC-AUC: {auc:.4f} {'✅' if auc > 0.75 else '⚠️'}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall: {recall:.4f}")
        print(f"    F1-score: {f1:.4f}")
        print("=== Résultats sur le TEST SET ===")
        print(f"ROC-AUC : {roc_auc_score(y_test, y_prob):.4f}")
        print(classification_report(y_test, y_pred, target_names=["Non converti", "Converti"]))

        # Matrice de confusion
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Non converti", "Converti"])
        plt.title("Matrice de confusion — Test Set")
        plt.savefig("confusion_matrix.png", dpi=150)
        plt.show()

    return results, trained_models

def evaluate_with_crossval(X_train, y_train):
    """
    We use StartifiedKFold to ensure we get the same class distribution.
    Cut training data into 5 equal folds(parts).
    For each model:
    - We test 5 times
    - Calculate the average score (avg_auc)
    - Calculate (std_auc)
    Pick the model with highest average score
    """
    models = get_models()

    cv = StratifiedKFold(
        n_splits=5, # 5 folds
        shuffle=True,   # mix data randomly before splitting
        random_state=42
    )

    print('\nCross Validation Results')
    print('#'*30)
    cv_results = {}

    for model_name, model in models.items():
        print(f'\n {model_name}')

        # here we test each model with 5 different splits       
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1
        )

        mean_auc = scores.mean()
        std_auc = scores.std()

        print(f"    Fold scores: {[round(s, 4) for s in scores]}")
        print(f"    Mean AUC: {mean_auc:.4f}    {'✅' if mean_auc > 0.75 else '⚠️'}")
        print(f"    Std AUC: {std_auc:.4f} {'stable' if std_auc > 0.05 else 'unstable'}")

        cv_results[model_name] = {
            'mean_auc': round(mean_auc, 4),
            'std_auc': round(std_auc, 4),
            'scores': scores
        }

        best_cv_model = max(cv_results, key=lambda m: cv_results[m]['mean_auc'])
        
        print(f'\nBest Model (Cross-Val): {best_cv_model}')
        print(f'Mean AUC: {cv_results[best_cv_model]['mean_auc']:.4f}')

        return cv_results, best_cv_model

def select_best_model(results, trained_models):
    """
    compares the models and selects the best one using ROC-AUC.
    which measures the model's ability to rank leads correctly.
    """

    print('\nModel Comparison Summary')
    print('#'*30)
    # Printing a table header
    print(f"\n{'Model':<25} {'ROC-AUC':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")

    for model_name, metrics in results.items():
        print(
            f"{model_name:<25}"
            f"{metrics['roc_auc']:>10.4f}"
            f"{metrics['precision']:>10.4f}"
            f"{metrics['recall']:>13.4f}"
            f"{metrics['f1_score']:>15.4f}"
        )

    # Choosing the max value based on the key of 'roc_auc'
    best_model_name = max(results, key=lambda m: results[m]['roc_auc'])
    best_model = trained_models[best_model_name]
    best_auc = results[best_model_name]['roc_auc']

    print(f"\n Best Model: {best_model_name}")
    print(f"    ROC-AUC : {best_auc:.4f}")

    return best_model_name, best_model

def probability_to_score(probability: float) -> float:
    """
    Converts the model's probability (0.0 - 1.0) to a score (0 - 100).
    """

    return round(probability * 100, 1)

def demonstrate_scoring(best_model, X_test, y_test):
    """
    Shows some leads with their predicted scores.
    """
    print("\nSample Lead Scores!")
    print('#'*30)

    # getting the proba of 'converted' for each leads
    probabilities = best_model.predict_proba(X_test)[:, 1] # result = [prob_n_c, prob_c]

    scores = [probability_to_score(p) for p in probabilities]

    print(f"\n{'Lead #':>8} {'Score':>8} {'Actual':>10} {'Priority':>15}")
    print("-"*45)

    # We add a condition to range the length of scores if were less than 10
    for i in range(min(10, len(scores))):
        score = scores[i]
        actual = y_test.iloc[i]

        # We define the priority for the lead
        if score >= 80:
            priority = '🔴 HIGH PRIORITY'
        elif score >= 50:
            priority = '🟡 MEDIUM'
        else:
            priority = '🟢 LOW'

        actual_label = 'Converted' if actual == 1 else 'Not converted'

        print(f"{i+1:>8} {score:>8.1f} {actual_label:>10} {priority:>15}")

def save_best_model(best_model, best_model_name):
    """
    We save the best trained model so we don't need to train it each time.
    """

    os.makedirs('models', exist_ok=True)
    model_path = 'models/best_model.pkl'

    # saving the file into the choosen path
    joblib.dump(best_model, model_path)

    print("\nBest Model saved!")
    print(f"Model: {best_model_name}")
    print(f"Path: {model_path}")

    # so we can know which one is the best model
    with open('models/best_model_name.txt', 'w') as f:
        f.write(best_model_name)

    return model_path

def explain_with_shap(best_model, X_train, X_test, feature_names):
    """
    SHAP is for explaining the model's predictions.
    It works by measuring each how much each feature affect on the final Score.
    it uses for parameters:
    - best_model : our trained model object
    - X_train : training data(It learns baseline from this)
    - X_test : test data(We explain predicition on this)
    - feature_names : list of column names 
    """

    print("\nSHAP Explainability analysis")
    print("#"*30)

    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # To create SHAP explainer, we use one for each model to learn 
    # the 'baseline' prediction from training data

    # returns the class name 'LogisticRegression','XGBClassifier'...
    model_type = type(best_model).__name__ 
    print(best_model)

    print(f"\n Model type detected: {model_type}")

    if model_type == 'LogisticRegression':
        explainer = shap.LinearExplainer(
            best_model, # the LinearRegression model
            X_train_df, # we use the training dataframe
            feature_perturbation='interventional' # Handles the connected features
        )
    else:
        # Used for Tree-based models. It's much faster and precis
        explainer = shap.TreeExplainer(best_model)

    # We use test set to calculate SHAP values
    # positive value = feature pushed prediction toward 'converted'
    # negative value = feature pushed prediction toward 'non converted'
    print("Calculating SHAP values...")
    raw = explainer.shap_values(X_test_df)
    if isinstance(raw, np.ndarray) and raw.ndim == 3:
        # Converting the shap from (X_test,features,classes) to (X_test,features)
        shap_values = raw[:, :, 1] 
    # Check if shap_values is a list
    elif isinstance(raw, list):
        # SHAP values can be a list of 2 arrays
        # [shap_for_class_0, shap_for_class_1]
        # we want converted class
        shap_values = raw[1]
    else:
        shap_values = raw
    
    print(" SHAP values calculated.")

    print("Global Feature Importance (Mean  | SHAP value|):")
    print("  " + "-"*50)

    mean_shap = np.abs(shap_values).mean(axis=0)

    shap_importance = pd.Series(mean_shap, index=feature_names)
    # Sorting the most 10 important shap  
    shap_importance_sorted = shap_importance.sort_values(ascending=False)

    for i, (feature, importance) in enumerate(shap_importance_sorted.head(10).items()):
        bar = "█" * int(importance * 100)
        print(f"    {i+1:2}. {feature:<30} {bar} {importance:.4f}")

    print("\n Individual Lead Explanation (Lead #1 from test set):")
    print("  " + "-"*50)

    # Get the shap value of the first lead
    lead_shap = shap_values[0]

    # Get the features of the first lead
    lead_features = X_test_df.iloc[0]

    explanation_df = pd.DataFrame({
        'Feature': feature_names,
        'Value': lead_features.values,
        'SHAP Impact': lead_shap
    })

    # Sort by the most SHAP that has impact
    explanation_df = explanation_df.reindex(
        explanation_df['SHAP Impact'].abs().sort_values(ascending=False).index
    )

    print(f"\n  {'Feature':<30}  {'Value':>8}  {'Impact':>10}  {'Direction':>12}")
    print("   " + "-"*65)

    
    for _, row in explanation_df.head(10).iterrows():
        # iterrows() loops through DataFrame rows as (index, Series) pairs
        # _ ignore the index
        direction = "↑ pushes UP" if row['SHAP Impact'] > 0 else "↓ pushes DOWN"
        print(
            f"  {row['Feature']:<30}"
            f"{row['Value']:>8.3f}"
            f"{row['SHAP Impact']:>10.4f}"
            f"  {direction}"
        )

    # Save SHAP importance to CSV
    os.makedirs('models', exist_ok=True)
    shap_importance_sorted.to_csv('models/shap_importance.csv')
    print(f"\n SHAP importance saved successfully.")

    return shap_values, shap_importance_sorted

def conformal_prediction(best_model, X_train, y_train, X_test, y_test):
    """
    
    """
    print("\nConformal Prediction == Confidence Intervals")
    print("#"*30)

    # cv='prefit' used to not retrain the model 
    mapie = MapieClassifier(estimator=best_model, cv='prefit')

    # We fit MAPIE on training data
    # It uses this data to learn how uncertain the model is
    mapie.fit(X_train, y_train)

    # alpha=0.10 means that we accept the resk of 10% of falling on the wrong interval
    y_pred, y_pset = mapie.predict(X_test, alpha=0.10)

    print(f"\n{'Lead':<6} {'True':>6} {'Pred':>6} {'Interval':>14} {'Certain?':>10}")
    print("   " + "-"*50)

    for i in range(min(10, len(X_test))):
        # Getting the real answer for converting or not 
        true_label = int(y_test.iloc[i])
        # Getting the model's prediction answer for converting or not
        pred_label = int(y_pred[i])

        # y_pset shape: (n_samples, n_classes, n_alpha)
        includes_0 = bool(y_pset[i, 0, 0]) # 'non converted' is included
        includes_1 = bool(y_pset[i, 1, 0]) # 'converted' is included

        if includes_0 and includes_1:
            interval = "{0, 1}"
            certain = "⚠️  No"
        elif includes_1:
            interval = "{1}"
            certain = "✅ Yes"
        elif includes_0:
            interval = "{0}"
            certain = "✅ Yes"
        else:
            interval = "{}"
            certain = "⚠️  No"

    print(f"    {i+1:<2} {true_label:>5} {pred_label:>5} {interval:>12} {certain:>12}")

    # We add 1 for each lead with True condition so we can calculate
    # the total number of certain predictions 
    certain_count = sum(
        1 for i in range(len(X_test))
        if y_pset[i, 0, 0] != y_pset[i, 1, 0]
    )

    print("Results:")
    print(f"    Total test leads: {len(X_test)}")
    print(f"    Certain predictions: {certain_count} ({certain_count/len(X_test)*100:.1f}%)")
    uncertain_count = len(X_test) - certain_count
    print(f"    Uncertain predictions: {uncertain_count} ({uncertain_count/len(X_test)*100:.1f}%)")
    print(f"\n  CONFORMAL PREDICTION COMPLETE.")

    return mapie, y_pred, y_pset

def track_with_mlflow(models_results, best_model, best_model_name,
                      X_train, y_train, X_test, y_test,
                      feature_names, scaler):
    
    print("\n" + "="*60)
    print("MLFLOW EXPERIMENT TRACKING")
    print("="*60)

    # We use a name for this experiment(container).
    # So all the infos will be saved under the name of it.
    mlflow.set_experiment("tasmim_lead_scoring")

    for model_name, results in models_results.items():
        
        # To make each run unique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{model_name}_{timestamp}"

        # This open a new session
        with mlflow.start_run(run_name=run_name):

            mlflow.set_tag("stage", "development")
            mlflow.set_tag("dataset_size", str(len(X_train) + len(X_test)))

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("n_features", len(feature_names))
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_param("random_state", 42)
            mlflow.log_param("scaler", "StandardScaler")
            
            model = results['model']
            # Get the model parameters as a dictionary
            model_params = model.get_params()
            for param_name, param_value in model_params.items():
                mlflow.log_param(f"model_{param_name}", str(param_value))

            mlflow.log_metric("ROC_AUC", results['roc_auc'])
            mlflow.log_metric("precision", results['precision'])
            mlflow.log_metric('Recall', results['recall'])
            mlflow.log_metric('F1_score', results['f1_score'])

            if model_name == best_model_name:
                mlflow.sklearn.log_model(
                    best_model,
                    artifact_path='best_model', # The folder where the model file stored with MLflow run.
                    registered_model_name='tasmim_lead_scorer' # to add the model to MLflow Model Registry
                )
                print(f"    Best Model saved: {model_name}")
                
                shap_path = 'models/shap_importance.csv'
                if os.path.exists(shap_path):
                    # Saves the file inside the MLflow run folder
                    mlflow.log_artifact(shap_path)
                print(" SHAP importance logged as artifact")
                
        
    print(f"\n MLflow Tracking Summary:")
    print(f"    Experiment: tasmim_lead_scoring")
    print(f"    Total runs: {len(models_results)}")
    print(f"    Best Model: {best_model_name}")
    print("MLflow tracking complete.")

    return models_results, best_model_name


if __name__ == '__main__':

    # Loading the features data from the csv file
    X, y = load_feature_matrix()

    # Split and scale the data
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    # Train all models and evaluate them
    results, trained_models = train_and_evaluate(
        X_train, X_test, y_train, y_test
    )

    # Use the Cross-Val Method
    cv_results, best_cv_model_name = evaluate_with_crossval(
        X_train, y_train
    )
    # Compare and select the best model
    best_model_name, best_model = select_best_model(results, trained_models)

    # Get feature names from the original DataFrame before scaling
    X_original, _ = load_feature_matrix()
    feature_names = list(X_original.columns)

    shap_values, shap_importance = explain_with_shap(
        best_model,
        X_train,
        X_test,
        feature_names
    )

    mapie, y_pred_conformal, y_pset = conformal_prediction(
        best_model,
        X_train, y_train, 
        X_test, y_test
    )

    track_with_mlflow(
        models_results  = results,
        best_model      = best_model,
        best_model_name = best_model_name,
        X_train         = X_train,
        y_train         = y_train,
        X_test          = X_test,
        y_test          = y_test,
        feature_names   = feature_names,
        scaler          = scaler
    )

    # Show some leads with their scores
    demonstrate_scoring(best_model, X_test, y_test)

    # Save the best model
    save_best_model(best_model, best_model_name)

    print(f"="*45)
    print(f"Best Model: {best_model_name}")
    print(f"Saved to: 'models/best_model.pkl")
    print(f"y_test distribution: {y_test.value_counts().to_dict()}")
