import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_connection import get_connection
from features.structred_features import get_structred_features
from features.nlp_features import get_nlp_features


def build_feature_matrix():
    """
    Here we rebuild the dataframe including
    the structred and nlp features additions.
    """

    conn = get_connection()
    df = pd.read_sql("SELECT * FROM leads", conn)
    conn.close()
    print(f"{len(df)} Leads loaded successfuly.")

    # Extracting the structred features we builded before
    df, structred_cols = get_structred_features(df)

    # Extracting the nlp features we builded before
    df, nlp_cols = get_nlp_features(df)

    # Combining them into one dataframe
    all_features_cols = structred_cols + nlp_cols

    # creating a copy of the features dataframe
    X = df[all_features_cols].copy()

    # creating a copy of the target variable
    # we're using the copy to not miss with the real dataframe
    y = df['converted'].copy()

    print(f" the features's shape: {X.shape[1]}")
    print(f" Number of leads: {y.shape[0]}")
    print(f"Converted = 1 : {(y==1).sum()} ({(y.mean())*100:.1f}%)")
    print(f"Converted = 0 : {(y==0).sum()} ({(1-y.mean())*100:.1f}%)")

    return X, y, all_features_cols

def check_data_quality(X, y):
    """
    Verifies the clean of data.
    We'll check the NaN values, infinite values,
    class balance, duplicate rows.
    """

    issues_found = 0

    # 1. Checking Missing Values
    missing = X.isnull().sum()
    
    # Keeping just the columns that have at least one missing value
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        print(f"Missing values found in: {list(missing_cols.index)}")
        X = X.fillna(X.median())
        print("Filling the missing values with the median.")
        issues_found += 1
    else:
        print("No missing values were found.")

    # 2. Checking infinite values
    """
    we'll check the infinite values because sometimes
    after the division operations we can get an infinite values 
    that can effect our operations.
    """
    # Selecting just the columns with numerical values
    inf_count = np.isinf(X.select_dtypes(include=np.number)).sum().sum()
    if inf_count > 0:
        print(f" {inf_count} infinite value were founded.")

        # replacing the infinite values with 0
        X = X.replace([np.inf, -np.inf], 0)
        print("Replacing infinite values done.")
        issues_found += 1
    else:
        print("No infinite values were founded.")

    # 3. Checking Class balance
    conversion_rate = y.mean()
    if conversion_rate < 0.1 or conversion_rate > 0.9:
        print(f"{conversion_rate * 100:.1f}% is not a balance class.")
        issues_found += 1
    else:
        print(f"{conversion_rate*100:.1f}% is an acceptable class balance")

    # 4. Checking duplicate rows
    dup_count = X.duplicated().sum()
    if dup_count > 0:
        print(f"{dup_count} duplicate rows detected.")

        # Initialiazing new X, y with no duplicated rows
        mask = ~X.duplicated()
        X = X[mask]
        y = y[mask]
        print("Duplicated rows removed.")
        issues_found += 1
    else:
        print("No duplicatd rows were founded.")
    
    if issues_found == 0:
        print("Our Data is perfectly clean.")
    else:
        print("Some issues were detected and fixed automatically.")
    return X, y

def analyze_features(X, y):
    """
    Analyze which features are the strongest and most predictive.
    We will analyze them by calculating the correlation.
    """

    analysis_df = X.copy()
    analysis_df['converted'] = y.values
    # i choosed to search the correlation with each feature
    # then delete the target column so we don't get always 1.0
    correlations = analysis_df.corr()['converted'].drop('converted')
    correlations_sorted = correlations.abs().sort_values(ascending=False)
    print("The Top 10 most predictive features:")
    print("   " + "-" * 45)

    # we used enumerate to have the index of i and the (feature, value) pair
    for i, (feature, corr_value) in enumerate(correlations_sorted.head(10).items()):
        bar_length = int(corr_value * 30)
        bar = "█" * bar_length

        print(f"   {i+1:2}. {feature:<30} {bar} {corr_value:.3f}")

    return correlations_sorted

def save_feature_matrix(X, y, feature_cols):
    """
    Save the final result to a csv file so we can work smoothly
    in the next step on notebook.
    """

    os.makedirs('data', exist_ok=True)
    final_df = X.copy()
    # we use values so we avoid the index issues
    final_df['converted'] = y.values
    output_path = 'data/feature_matrix.csv'
    # Saving the file in that specific path and ignoring the index
    final_df.to_csv(output_path, index=False)
    print("Feature File saved successfully.")
    print(f"Columns: {list(final_df.columns)}")

    return output_path

if __name__ == '__main__':
    X, y, feature_cols = build_feature_matrix()
    X, y = check_data_quality(X, y)
    correlations = analyze_features(X, y)
    path = save_feature_matrix(X, y, feature_cols)