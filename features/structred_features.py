import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_connection import get_connection

def load_data():

   # load all the leads data from postgreSQL into a Pandas Dataframe.
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM leads", conn)
    conn.close()
    print("Loading completed successfuly.")
    return df

def encode_ordinal_features(df):
    """
    We use Ordinal encoding by giving ordinal values. 
    """

    budget_order = {'low': 0, 'medium': 1, 'high': 2, 'enterprise': 3}
    df['budget_encoded'] = df['budget_range'].map(budget_order)

    size_order = {'solo': 0, 'small': 1, 'medium': 2, 'large': 3}
    df['company_size_encoded'] = df['company_size'].map(size_order)

    deadline_order = {'flexible': 0, 'normal': 1, 'urgent': 2}
    df['deadline_encoded'] = df['deadline'].map(deadline_order)

    return df

def encode_categorical_features(df):
    """
    we use One-hot encode to create a separate column for each option.
    service_type = 'website'
    becomes:
    service_website=1, service_ecommerce=0, service_app=0, 
    service_branding=0, service_seo=0
    ...
    """   

    service_dummies = pd.get_dummies(
        df['service_type'],
        prefix='service'  # column names become: service_website, service_ecommerce...
    )
    
    channel_dummies = pd.get_dummies(
        df['contact_channel'],
        prefix='contact'
    )

    df = pd.concat([df, service_dummies, channel_dummies], axis=1)

    print(f"service columns: {list(service_dummies.columns)}")
    print(f"channel columns: {list(channel_dummies.columns)}")
    return df

def extract_time_features(df):
    """
    The timimg of the inquiry can be a signal
    for the intersing of the client.

    - Monday morning = serious, professional
    - Sunday midnight = probably not serious
    ...

    we extract useful time features from created_at.
    """

    # Make sure we are working the column is treated as a datetime
    df['created_at'] = pd.to_datetime(df['created_at'])

    # Extracting the hour of the day (0-23)
    df['hour_of_day'] = df['created_at'].dt.hour

    # Extracting the day of the week (0-6)
    df['day_of_week'] = df['created_at'].dt.dayofweek

    # Extracting the month of the year (1-12)
    df['month_of_year'] = df['created_at'].dt.month

    # Checking if the inquiry created in a business hour,day (9AM - 6PM)
    df['is_business_hours'] = (
        (df['hour_of_day'] >= 9) &
        (df['hour_of_day'] <= 18) &
        (df['day_of_week'] < 5 )
    ).astype(int) # Giving a True/False result = 1/0

    print("Time features extracted.")
    return df

def get_structred_features(df):
    """
    Running all the previous features in order
    and returns only the final numerical columns,
    because the ML model that what is going to receive.
    """

    df = encode_ordinal_features(df)
    df = encode_categorical_features(df)
    df = extract_time_features(df)

    # Selecting just the numerical columns
    feature_columns = [
        'budget_encoded',
        'company_size_encoded',
        'deadline_encoded',
        'service_website',
        'service_ecommerce',
        'service_mobile_app',
        'service_branding',
        'service_seo',
        'contact_email',
        'contact_whatsapp',
        'contact_phone',
        'contact_social_media',
        'hour_of_day',
        'day_of_week',
        'month_of_year',
        'is_business_hours'
    ]

    # Keeping just the columns in list in case if some hidden of forgot columns were founded
    existing_columns = [c for c in feature_columns if c in df.columns]

    print(f"\n structred features matrix shape: {df[existing_columns].shape}")
    print(f"{len(existing_columns)} features extracted from structred data")
    return df, existing_columns

# Main — test the file directly

if __name__ == '__main__':
    df = load_data()
    df, feature_cols = get_structred_features(df)

    print(df[feature_cols].head())
    print('\n')
    print(df[feature_cols].isnull().sum())
