import pandas as pd
import numpy as np
import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_connection import get_connection
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

sia = SentimentIntensityAnalyzer()
print("NLP packages loaded.")

def extract_text_feature(df):

    """ Extract some simple features from the message"""

    # Count the number of words for each message
    df['message_length'] = df['message_text'].apply(
        lambda x: len(str(x).split())
    )

    # we use this function because usually when we have bigger words that means that the words is professional
    def avg_word_len(text):
        words = str(text).split()
        if len(words) == 0:
            return 0
        else:
            return np.mean([len(word) for word in words])
        
    df['avg_word_length'] = df['message_text'].apply(avg_word_len)

    # Here we count the number of sentences by spliting the message by [. - ? - !]
    df['sentence_count'] = df['message_text'].apply(
        lambda x: len(re.split(r'[.!?]+', str(x)))
    )

    # Exclamation signal usually means something special
    df['exclamation_count'] = df['message_text'].apply(
        lambda x: str(x).count('!')
    )

    df['question_count'] = df['message_text'].apply(
        lambda x: str(x).count('?')
    )

    print("Simple Extracation done.")
    return df

def extract_pattern_features(df):
    """
    Here we are going to use some regex to detect specific patterns in the message
    """

    # Looking if budget is mentioned
    # Searching for patterns like(40000 MAD, 50K, budget, $, €)
    budget_pattern = r'(\d[\d\s]*(?:mad|dh|dirham|€|$|euro|budget|k\b) |\bbudget\b)'
    df['has_budget_mentioned'] = df['message_text'].apply(
        lambda x: 1 if re.search(budget_pattern, str(x).lower()) else 0
    )

    # Looking if deadline/date is mentioned
    # Searching for patterns like(months, urgent, deadline, delai, specific dates)
    deadline_pattern = r'(january|february|march|april|may|june|july|august|september|october|november|december|urgent|deadline|timeline|week|month|days|\d{1,2}/\d{1,2})'
    df['has_deadline_mentioned'] = df['message_text'].apply(
        lambda x: 1 if re.search(deadline_pattern, str(x).lower()) else 0
    )

    # Looking if phone number is found because it is a strong point that refers that the client is intersted
    # We'll search on the most used ways to write a phone number like(06... - 07.... - 05... - +212...)
    phone_pattern = r'(\+212|0[567])[\s\-]?\d{2}[\s\-]?\d{2}[\s\-]?\d{2}[\s\-]?\d{2}'
    df['has_phone_mentioned'] = df['message_text'].apply(
        lambda x: 1 if re.search(phone_pattern, str(x).lower()) else 0
    )

    # Looking if the organisation is mentioned like(company,entreprise,startup...)
    company_pattern = r'(notre (société|entreprise|startup|company|agence)|nous sommes)'
    df['has_company_mentioned'] = df['message_text'].apply(
        lambda x: 1 if re.search(company_pattern, str(x).lower()) else 0
    )

    print('Loading patterns features done.')
    print(f"{df['has_budget_mentioned'].sum()} is the number of leads that mentioned budget.")
    print(f"{df['has_deadline_mentioned'].sum()} is the number of leads that mentioned deadline.")
    print(f"{df['has_phone_mentioned'].sum()} is the number of leads that mentioned phone number.")
    print(f"{df['has_company_mentioned'].sum()} is the number of leads that mentioned company.")
    return df

def extract_professionalism_score(df):
    """
    1. we define 2 lists of keywords:
    - Professional vocabulary = increase score
    - Informal vocabulary = decrease score

    2. we calculate the ratio on how many professional words appear in the message.
    """

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
        'plz', 'pls', 'thx', 'lol', 'omg', 'wtf', 'idk',
    'wanna', 'gonna', 'gimme', 'hey', 'yo', 'sup', 'ngl'
    ]

    def calculate_score(text):
        text_lower = str(text).lower()
        words = text_lower.split()
        if len(words) == 0:
            return 0.0
        
        # Count how many professional and informal words in each message
        pro_count = sum(1 for word in words if word in professional_keywords)
        informal_count = sum(1 for word in words if word in informal_keywords)

        # Caculate % of the professional words from all the words
        pro_ratio = pro_count/len(words)

        # Gives a penalty for informal words
        informal_penalty = informal_count * 0.1

        # Gives a Bonus for longer messages and we stop the bonus score for longer messages on 0.3
        length_bonus = min(len(words) / 100, 0.3)

        score = pro_ratio + length_bonus - informal_penalty

        # we use Clamping = stoping the score from growing than 1.0 and reducing than 0.0
        return max(0.0, min(1.0, score))
    
    df['professionalism_score'] = df['message_text'].apply(calculate_score)

    print("Professionalism score calculated.")
    print(f"Average score: {df['professionalism_score'].mean():.3f}")
    return df

def extract_sentiment_score(df):
    """
    Sentiment analysis mesures the emotional tone of the text.
    Positive sentiment = Motivated, Excited Client
    Negative sentiment = frustrated, dountful Client

    We use VADER - a sentiment analyzer that works well
    even without training data. It was built for short texts.
    """

    def get_sentiment(text):
        scores = sia.polarity_scores(str(text))
        # compound score ranges between -1 (very negative) and 1 (very positive)
        # we could range it between 0-1
        return (scores['compound'] + 1) / 2
    
    df['sentiment_score'] = df['message_text'].apply(get_sentiment)

    print("Sentiment score saved successfuly.")
    print(f"Average Sentiment {df['sentiment_score'].mean():.3f}")
    return df

def get_nlp_features(df):
    """
    Runs all NLP extractions in order and returns
    the list of NLP feature columns names.
    """

    df = extract_text_feature(df)
    df = extract_pattern_features(df)
    df = extract_professionalism_score(df)
    df = extract_sentiment_score(df)

    nlp_feature_columns = [
        'message_length',
        'avg_word_length',
        'sentence_count',
        'exclamation_count',
        'question_count',
        'has_budget_mentioned',
        'has_deadline_mentioned',
        'has_phone_mentioned',
        'has_company_mentioned',
        'professionalism_score',
        'sentiment_score'
    ]

    print(f"\n NLP feature matrix shape: {df[nlp_feature_columns].shape}")
    print(f"{len(nlp_feature_columns)} NLP features extracted")

    return df, nlp_feature_columns

# Main — test the file directly

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from database.db_connection import get_connection

    conn = get_connection()
    df = pd.read_sql("SELECT * FROM leads", conn)
    conn.close()

    df, nlp_cols = get_nlp_features(df)
    
    print(df[nlp_cols + ['converted']].head(10))
    correlations = df[nlp_cols + ['converted']].corr()['converted'].drop('converted')
    print(correlations.sort_values(ascending=True))