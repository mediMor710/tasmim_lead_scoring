import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import joblib
import numpy as np
import pandas as pd
import re
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from datetime import datetime
from database.db_connection import get_connection

sia = SentimentIntensityAnalyzer()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'base_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
NAMES_PATH = os.path.join(BASE_DIR, 'models', 'feature_names.json')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open (NAMES_PATH, 'r') as f:
    FEATURE_NAMES = json.load(f)

print(f"Pipeline tasks ready - model: {type(model).__name__}")

def task_extract_leads(**context):
    """
    Task 1: Find all leads that have no score yet.

    '**context' sends information about the task.
    """
    print("#"*50)
    print("\nTASK 1 - Extracting unscored leads")
    print("#"*50)

    conn = get_connection()

    query = """
        SELECT l.id from leads l
        LEFT JOIN scores s ON l.id = s.lead_id
        WHERE s.id IS NULL            
    """
    df = pd.read_sql(query, conn)
    conn.close()

    lead_ids = df['id'].tolist()

    print(f"Found {len(lead_ids)} unscored leads")

    if len(lead_ids) == 0:
        print("All leads already scored - nothing to do.")

    # Pushing the lead_ids so the next task can pull them by the same 'key'
    context['ti'].xcom_push(key='lead_ids', value=lead_ids)
    return len(lead_ids)

def task_build_features(**context):
    """
    Task 2: Load full lead data and 
    build features for each unscored lead.
    """

    print("#"*50)
    print("TASK 2 - Building features")
    print("#"*50)

    lead_ids = context['ti'].xcom_pull(task_ids='extract_leads', key='lead_ids')

    if not lead_ids:
        print("No leads to process.")
        context['ti'].xcom_push(key='features', value=[])
        return 0
    
    conn = get_connection()
    query = f"""
    Select * from leads
    Where id in ({','.join(map(str, lead_ids))})
    """
    df = pd.read_sql(query, conn)
    conn.close()

    print(f"Processing {len(df)} leads...")

    features_list = []
    # Process each lead with ignoring the index
    for _, row in df.iterrows():
        lead_dict = row.to_dict()
        features = _build_one_lead_features(lead_dict)
        features['lead_id'] = lead_dict['id']
        features_list.append(features)

    print(f"Features built for {len(features_list)} leads")

    context['ti'].xcom_push(key='features', value=features_list)
    return len(features_list)

def _build_one_lead_features(lead_data: dict) -> dict:
    """
    Helper — builds the feature dict for a single lead.
    Exact same logic as predictor.py's build_features_from_lead().
    """

    row = {}

    # Ordinal encoding
    budget_order   = {'low': 0, 'medium': 1, 'high': 2, 'enterprise': 3}
    size_order     = {'solo': 0, 'small': 1, 'medium': 2, 'large': 3}
    deadline_order = {'flexible': 0, 'normal': 1, 'urgent': 2}

    row['budget_encoded']       = budget_order.get(lead_data.get('budget_range', 'low'), 0)
    row['company_size_encoded'] = size_order.get(lead_data.get('company_size', 'solo'), 0)
    row['deadline_encoded']     = deadline_order.get(lead_data.get('deadline', 'normal'), 0)

    # One-hot encoding
    for s in ['website', 'ecommerce', 'mobile_app', 'branding', 'seo']:
        row[f'service_{s}'] = 1 if lead_data.get('service_type') == s else 0

    for c in ['email', 'whatsapp', 'phone', 'social_media']:
        row[f'contact_{c}'] = 1 if lead_data.get('contact_channel') == c else 0

    # Time features
    created_at = pd.to_datetime(lead_data.get('created_at', datetime.now()))
    row['hour_of_day']       = created_at.hour
    row['day_of_week']       = created_at.weekday()
    row['month_of_year']     = created_at.month
    row['is_business_hours'] = 1 if (9 <= created_at.hour <= 18 and created_at.weekday() < 5) else 0

    # NLP features
    text  = str(lead_data.get('message_text', ''))
    words = text.split()

    row['message_length']    = len(words)
    row['avg_word_length']   = np.mean([len(w) for w in words]) if words else 0
    row['sentence_count']    = len(re.split(r'[.!?]+', text))
    row['exclamation_count'] = text.count('!')
    row['question_count']    = text.count('?')

    budget_pat   = r'(\d[\d\s]*(?:mad|dh|dirham|€|\$|euro|budget|k\b)|\bbudget\b)'
    deadline_pat = r'(january|february|march|april|may|june|july|august|september|october|november|december|urgent|deadline|timeline|week|month|\d{1,2}/\d{1,2})'
    phone_pat    = r'(\+212|0[67])[\s\-]?\d{2}[\s\-]?\d{2}[\s\-]?\d{2}[\s\-]?\d{2}'
    company_pat  = r'(our (company|business|startup|firm|agency)|we are)'

    row['has_budget_mentioned']   = 1 if re.search(budget_pat,   text.lower()) else 0
    row['has_deadline_mentioned'] = 1 if re.search(deadline_pat, text.lower()) else 0
    row['has_phone_mentioned']    = 1 if re.search(phone_pat,    text)         else 0
    row['has_company_mentioned']  = 1 if re.search(company_pat,  text.lower()) else 0

    professional_keywords = [
        'company', 'business', 'enterprise', 'director', 'manager', 'team',
        'project', 'development', 'integration', 'platform', 'solution',
        'ecommerce', 'application', 'mobile', 'crm', 'dashboard',
        'analytics', 'api', 'backend', 'frontend', 'portfolio',
        'proposal', 'meeting', 'availability', 'timeline', 'deadline',
        'budget', 'investment', 'objective', 'strategy', 'requirements',
        'hello', 'regards', 'sincerely', 'please', 'kindly'
    ]
    informal_keywords = ['plz', 'pls', 'thx', 'lol', 'omg', 'idk',
                         'wanna', 'gonna', 'gimme', 'hey', 'yo', 'sup']

    word_list    = text.lower().split()
    pro_count    = sum(1 for w in word_list if w in professional_keywords)
    inf_count    = sum(1 for w in word_list if w in informal_keywords)
    pro_ratio    = pro_count / len(word_list) if word_list else 0
    length_bonus = min(len(word_list) / 100, 0.3)
    row['professionalism_score'] = max(0.0, min(1.0, pro_ratio + length_bonus - inf_count * 0.1))

    compound = sia.polarity_scores(text)['compound']
    row['sentiment_score'] = (compound + 1) / 2

    return row

def task_score_and_save(**context):
    """
    Task 3: Score all leads and save results to the scores table.
    """

    print("#"*30)
    print("TASK 3 - Scoring and saving")
    print("#"*30)

    features_list = context['ti'].xcom_pull(task_ids='build_features', key='features')

    if not features_list:
        print("No features to score.")
        return 0
    
    for f in features_list:
        lead_ids = f.pop('lead_id')
    
    X = pd.DataFrame(features_list).reindex(columns=FEATURE_NAMES, fill_value=0)
    X_scaled = scaler.transform(X.values)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    for p in probabilities:
        scores = round(float(p) * 100, 1)

    conn = get_connection()
    cursor = conn.cursor()

    insert_query = """
        INSERT INTO scores (lead_id, score, scored_at)
        VALUES (%s, %s, %s)
    """
    scored_at = datetime.now()
    for lead_id, score in zip(lead_ids, scores):
        cursor.execute(insert_query, (lead_id, score, scored_at))

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Saved {len(scores)} scores to database")
    print(f"   High priority (≥75): {sum(1 for s in scores if s >= 75)}")
    print(f"   Medium (45–74)      : {sum(1 for s in scores if 45 <= s < 75)}")
    print(f"   Low (<45)           : {sum(1 for s in scores if s < 45)}")

    return len(scores)