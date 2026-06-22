import joblib
import numpy as np
import pandas as pd
import json
import os
import re
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nltk
nltk.download('vader_lexicon', quiet=True)  # Hiding the download messages
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime

sia = SentimentIntensityAnalyzer()

##### PART A: Load model and scaler once the API starts #####
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
SHAP_PATH = os.path.join(BASE_DIR, 'models', 'shap_importance.csv')
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, 'models', 'feature_names.json')
print(FEATURE_NAMES_PATH)

with open(FEATURE_NAMES_PATH, 'r') as f:
    FEATURE_NAMES = json.load(f)
print(f"Feature order loaded: {len(FEATURE_NAMES)} features")

print('Loading model and scaler...')
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Load feature names from SHAP importance file
# 'index_col=0' Makes the first column as the index
shap_df = pd.read_csv(SHAP_PATH, index_col=0) 
FEATURE_NAMES = list(shap_df.index)

print(f"Model Loaded: {type(model).__name__}")
print(f"Feature expected: {len(FEATURE_NAMES)}")

##### Build Features #####
def build_features_from_lead(lead_data: dict) -> pd.DataFrame:
    """
    Takes a raw lead dictionary and returns a single row DataFrame with 
    all 27 features the model expects.
    """

    row = {}

    # Ordinal encoding
    budget_order = {'low': 0, 'medium': 1, 'high': 2, 'enterprise': 3}
    size_order = {'solo': 0, 'small': 1, 'medium': 2, 'large': 3}
    deadline_order = {'flexible': 0, 'normal': 1, 'urgent': 2}

    # Converting from categorical values into ordinal values
    row['budget_encoded'] = budget_order[lead_data['budget_range']]
    row['company_size_encoded'] = size_order[lead_data['company_size']]
    row['deadline_encoded'] = deadline_order[lead_data['deadline']]

    # One-hot encoding 
    for s in ['website', 'ecommerce', 'mobile_app', 'branding', 'seo']:
        row[f'service_{s}'] = 1 if lead_data['service_type'] == s else 0

    for c in ['email', 'whatsapp', 'phone', 'social_media']:
        row[f'contact_{c}'] = 1 if lead_data['contact_channel'] == c else 0

    now = datetime.now()
    row['hour_of_day'] = now.hour
    row['day_of_week'] = now.weekday()
    row['month_of_year'] = now.month
    row['is_business_hours'] = 1 if (9 <= now.hour <= 18 and now.weekday() < 5) else 0

    text = str(lead_data['message_text'])
    words = text.split()

    row['message_length'] = len(words)
    row['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
    row['sentence_count'] = len(re.split(r'[.!?]+', text))
    row['exclamation_count'] = text.count('!')
    row['question_count'] = text.count('?')

    # Looking for specific patterns with Regex
    budget_pat = r'(\d[\d\s]*(?:mad|dh|dirham|€|\$|euro|budget|k\b)|\bbudget\b)'
    deadline_pat = r'(january|february|march|april|may|june|july|august|september|october|november|december|urgent|deadline|timeline|week|month|\d{1,2}/\d{1,2})'
    phone_pat = r'(\+212|0[567])[\s\-]?\d{2}[\s\-]?\d{2}[\s\-]?\d{2}[\s\-]?\d{2}'
    company_pat = r'(our (company|business|startup|firm|agency)|we are)'

    row['has_budget_mentioned'] = 1 if re.search(budget_pat, text.lower()) else 0
    row['has_deadline_mentioned'] = 1 if re.search(deadline_pat, text.lower()) else 0
    row['has_phone_mentioned'] = 1 if re.search(phone_pat, text) else 0
    row['has_company_mentioned'] = 1 if re.search(company_pat, text.lower()) else 0

    # Assigning some professionalism keywords
    professional_keywords = [
        'company', 'business', 'enterprise', 'director', 'manager', 'team',
        'project', 'development', 'integration', 'platform', 'solution',
        'ecommerce', 'application', 'mobile', 'crm', 'dashboard',
        'analytics', 'api', 'backend', 'frontend', 'portfolio',
        'proposal', 'meeting', 'availability', 'timeline', 'deadline',
        'budget', 'investment', 'objective', 'strategy', 'requirements',
        'hello', 'regards', 'sincerely', 'please', 'kindly'
    ]
    informal_keywords = [
        'plz', 'pls', 'thx', 'lol', 'omg', 'idk',
        'wanna', 'gonna', 'gimme', 'hey', 'yo', 'sup'
    ]

    text_lower = text.lower()
    word_list = text_lower.split()
    pro_count = 0
    inf_count = 0
    for w in word_list:
        if w in professional_keywords:
            pro_count += 1
        elif w in informal_keywords:
            inf_count += 1
    pro_ratio = pro_count / len(word_list) if word_list else 0
    length_bonus = min(len(word_list) / 100, 0.3)
    row['professionalism_score'] = max(0.0, min(1.0, pro_ratio + length_bonus - inf_count * 0.1))

    # Sentiment score (Vader)
    compound = sia.polarity_scores(text)['compound']
    row['sentiment_score'] = (compound + 1) / 2

    df = pd.DataFrame([row])
    # if there wasn't any column in FEATURE_NAMES it will assign it and fill it with 0s
    df = df.reindex(columns=FEATURE_NAMES, fill_value=0)

    return df

##### C: Score a Lead #####
def score_lead(lead_data: dict) -> dict:
    """
    Takes a raw lead data and returns a full scoring result.
    """

    X = build_features_from_lead(lead_data)
    X_scaled = scaler.transform(X.values)

    # Predict probability of converted class
    prob = model.predict_proba(X_scaled)[0][1]

    score = round(prob * 100, 1)

    if score >= 75:
        priority = 'HIGH'
        recommendation = 'Contact this lead immediatly - high conversion potential.'
    elif score >= 45:
        priority = 'MEDIUM'
        recommendation = 'Follow up within 48 hours - worth pursuing.'
    else:
        priority = 'LOW'
        recommendation = 'Send automated response - low conversion potential'

    # Display the top features that assists to grow up the score
    if '0' in shap_df.columns:
        top_factors = shap_df.head(5)['0'].to_dict()
    else:
        # returning all the rows from the first column
        top_factors = shap_df.head(5).iloc[: , 0].to_dict()

    return {
        'full_name': lead_data['full_name'],
        'score': score,
        'priority': priority,
        'conversion_probability': round(float(prob), 4),
        'top_factors': top_factors,
        'recommendation': recommendation
    }