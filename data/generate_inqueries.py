import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
from database.db_connection import get_connection
from data.paraphrasing import paraphrase, validate_paraphrase

fake = Faker('en-Us')  # Gives fake english data
random.seed(42)
np.random.seed(42)     # Same seed for numpy

# Defining ready values of categories

COMPANY_SIZES = ['solo', 'small', 'medium', 'large']
SERVICE_TYPES = ['website', 'ecommerce', 'mobile_app', 'branding', 'seo']
BUDGET_RANGES = ['low', 'medium', 'high', 'enterprise']
DEADLINES     = ['urgent', 'normal', 'flexible']
CHANNELS      = ['email', 'whatsapp', 'phone', 'social_media']

# Defining ready messages based on the category

PROFESSIONAL_MESSAGES = [
    "Hello, we are a distribution company based in Casablanca. We are looking to develop a full e-commerce platform with inventory management and online payment integration. Our budget is around 40,000 MAD and we would like to start in January. Could you send us a detailed proposal?",
    "Our company is looking to redesign our corporate website. We need a modern design, CRM integration, and an analytics dashboard. We have a team of 50 employees and need the project completed within 3 months.",
    "We are developing a mobile delivery management application. We are looking for an agency experienced in React Native. Available budget: 60,000 MAD. Timeline: 3 months. Please share your portfolio.",
    "I am the sales director at the Company. We need a complete digital overhaul including website, SEO, and social media management. When are you available for a meeting to discuss the project scope?",
    "We are a mid-sized retail chain looking to launch an online store. We currently have 12 physical locations and need a scalable e-commerce solution. Budget is flexible for the right partner.",
    "Our startup is seeking a development agency to build our MVP. We have detailed technical specifications ready. Timeline is urgent — we need to launch before Q2. Please confirm your availability.",
    "I represent a real estate firm in Casablanca. We need a modern property listing website with search filters, virtual tours, and a contact management system. We have a serious budget allocated for this project.",
    "We are looking for a long-term digital partner to handle our website maintenance, SEO strategy, and quarterly campaign landing pages. Our company has worked with agencies before and values quality communication.",
]

AVERAGE_MESSAGES = [
    "Hello, I would like to create a website for my business. Can you give me your pricing?",
    "Hi, I need a website for my restaurant. How much does it cost?",
    "Hello, we are looking for someone to manage our Instagram page and build a simple website for us.",
    "I want to launch an online store to sell clothes. Do you handle that kind of project?",
    "We are a small company and need to improve our online presence. What services do you offer?",
    "Hello, I saw your work online and I am interested. Can we talk about building a website for my shop?",
    "I need a professional website for my consulting business. Nothing too complicated, just something clean and modern.",
    "Our association needs a website to present our activities and collect donations online. What would that cost?",
]

VAGUE_MESSAGES = [
   "Hello I want a website",
    "How much does a website cost?",
    "I need help with the internet",
    "hi, website please",
    "hello info",
    "need a site",
    "website how much",
    "hello can you help",
]



# ─────────────────────────────────────────────
# STEP A: T5 Paraphrase
# ─────────────────────────────────────────────



def generate_lead():
    """Generates a single realistic lead dictionary.
    we define first 3 profile types of the clients.
    - Profile A: Serious prospect
    - Profile B: Average prospect (medium conversion probability)
    - Profile C: Window shopper 
    """

    # Most prospects are window shoppers
    profile = random.choices(
        ['serious', 'average', 'window_shopper'],
        weights = [0.25, 0.35, 0.40]
    )[0]

    if profile == 'serious':

        company_size  = random.choices(
            ['solo', 'small', 'medium', 'large'],
            weights=[0.05, 0.15, 0.40, 0.40]
        )[0]

        budget_range  = random.choices(
            ['low', 'medium', 'high', 'enterprise'],
            weights=[0.05, 0.15, 0.40, 0.40]
        )[0]

        deadline = random.choices(
            ['urgent', 'normal', 'flexible'],
            weights=[0.50, 0.40, 0.10]
        )[0]

        channel = random.choices(
            ['email', 'whatsapp', 'phone', 'social_media'],
            weights=[0.30, 0.35, 0.25, 0.10]
        )[0]

        # Pick message quality and actual message text
        message_quality = random.choices(
            ['professional', 'average', 'vague'],
            weights=[0.75, 0.20, 0.05]  # 75% professional, 20% average, 5% vague
        )[0]

    elif profile == 'average':

        company_size  = random.choices(
            ['solo', 'small', 'medium', 'large'],
            weights=[0.20, 0.40, 0.30, 0.10]
        )[0]

        budget_range  = random.choices(
            ['low', 'medium', 'high', 'enterprise'],
            weights=[0.20, 0.50, 0.25, 0.05]
        )[0]

        deadline = random.choices(
            ['urgent', 'normal', 'flexible'],
            weights=[0.25, 0.65, 0.15]
        )[0]

        channel = random.choices(
            ['email', 'whatsapp', 'phone', 'social_media'],
            weights=[0.30, 0.40, 0.15, 0.15]
        )[0]

        # Pick message quality and actual message text
        message_quality = random.choices(
            ['professional', 'average', 'vague'],
            weights=[0.20, 0.65, 0.15]
        )[0]

    else:

        company_size  = random.choices(
            ['solo', 'small', 'medium', 'large'],
            weights=[0.60, 0.30, 0.08, 0.02]
        )[0]

        budget_range  = random.choices(
            ['low', 'medium', 'high', 'enterprise'],
            weights=[0.70, 0.25, 0.04, 0.01]
        )[0]

        deadline = random.choices(
            ['urgent', 'normal', 'flexible'],
            weights=[0.05, 0.35, 0.60]
        )[0]

        channel = random.choices(
            ['email', 'whatsapp', 'phone', 'social_media'],
            weights=[0.25, 0.45, 0.10, 0.20]
        )[0]

        # Pick message quality and actual message text
        message_quality = random.choices(
            ['professional', 'average', 'vague'],
            weights=[0.05, 0.25, 0.70]  
        )[0]

    # Service type stays random for all profiles because it doesn't effect
    service_type = random.choice(SERVICE_TYPES)

    # Pick message based on quality
    if message_quality == 'professional':
        base_message = random.choice(PROFESSIONAL_MESSAGES)
    elif message_quality == 'average':
        base_message = random.choice(AVERAGE_MESSAGES)
    else:
        base_message = random.choice(VAGUE_MESSAGES)

    try:
        message = paraphrase(base_message) 
        message = validate_paraphrase(message, base_message)
    except Exception as e:
        print(f"Paraphrase failed: {e}")
        message = base_message + f" [ref-{random.randint(10000,99999)}]"

    if profile == 'serious':
        converted = 1 if random.random() < 0.82 else 0
    elif profile == 'average':
        converted = 1 if random.random() < 0.28 else 0
    else:
        converted = 1 if random.random() < 0.04 else 0

    # Generate a random date in the last 18 months
    days_ago = random.randint(1, 540)
    created_at = datetime.now() - timedelta(days=days_ago)

    return {
        'full_name':       fake.name(),
        'email':           fake.email(),
        'company_name':    fake.company(),
        'company_size':    company_size,
        'service_type':    service_type,
        'budget_range':    budget_range,
        'deadline':        deadline,
        'contact_channel': channel,
        'message_text':    message,
        'created_at':      created_at,
        'converted':       converted
    }


# ─────────────────────────────────────────────
# STEP E: Insert leads into PostgreSQL
# ─────────────────────────────────────────────

def insert_leads(leads: list):
    """
    Takes a list of lead dicts and inserts them into the database.
    """
    
    conn = get_connection()
    cursor = conn.cursor()

    insert_query = """
        INSERT INTO leads (
            full_name, email, company_name, company_size,
            service_type, budget_range, deadline,
            contact_channel, message_text, created_at, converted
        ) VALUES (
            %(full_name)s, %(email)s, %(company_name)s, %(company_size)s,
            %(service_type)s, %(budget_range)s, %(deadline)s,
            %(contact_channel)s, %(message_text)s, %(created_at)s, %(converted)s
        )
    """

    for lead in leads:
        cursor.execute(insert_query, lead)

    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ Successfully inserted {len(leads)} leads into the database.")