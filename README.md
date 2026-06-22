# 🎯 Tasmim Web — Intelligent Lead Scoring System

A machine learning system that automatically scores incoming leads for Tasmim Web
and predicts which ones are most likely to become real clients.

---

## 📋 What It Does

When a potential client contacts Tasmim Web, this system:

1. Receives the lead (via Google Form or API)
2. Analyzes the message, budget, company size, deadline, and more
3. Assigns a score from **0 to 100**
4. Labels it as 🔴 HIGH / 🟡 MEDIUM / 🟢 LOW priority
5. Saves the score to the database automatically every night

---

## 🗂️ Project Structure

```
tasmim_lead_scoring/
│
├── data/
│   ├── generate_leads.py       → generates synthetic leads for training
│   ├── sync_from_sheets.py     → syncs new leads from Google Forms
│   ├── score_all_leads.py      → scores all unscored leads at once
│   └── feature_matrix.csv      → saved feature matrix (auto-generated)
│
├── database/
│   └── db_connection.py        → PostgreSQL connection
│
├── features/
│   ├── structured_features.py  → encodes budget, size, deadline, channel
│   ├── nlp_features.py         → extracts NLP features from message text
│   └── build_features.py       → combines all features into one matrix
│
├── models/
│   ├── train_model.py          → trains and evaluates 3 ML models
│   ├── best_model.pkl          → saved best model (auto-generated)
│   ├── scaler.pkl              → saved scaler (auto-generated)
│   ├── feature_names.json      → saved feature order (auto-generated)
│   └── shap_importance.csv     → SHAP feature importance (auto-generated)
│
├── API/
│   ├── main.py                 → FastAPI endpoints
│   ├── schema.py               → request/response data shapes
│   └── predictor.py            → loads model and scores a lead
│
├── Airflow/
│   ├── scoring_pipeline.py     → DAG definition (schedule & task order)
│   └── pipeline_tasks.py       → the 3 pipeline task functions
│
├── interface/
│   ├── app.py                  → home page (metrics + leads table)
│   └── pages/
│       ├── charts.py         → charts & analytics page
│       └── score_lead.py     → score a new lead from Google Forms
│
├── notebooks/
│   └── generate_data.ipynb     → Jupyter notebook for data generation
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the project
```bash
git clone https://github.com/mediMor710/tasmim_lead_scoring.git
cd tasmim_lead_scoring
```

### 2. Install all dependencies
```bash
pip install -r requirements.txt
```

### 3. Install PostgreSQL inside WSL
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib -y
sudo service postgresql start
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'your_password_here';"
```

### 4. Create the database
```bash
sudo -u postgres psql
```

Then inside psql:
```sql
CREATE DATABASE tasmim_leads;

\c tasmim_leads

CREATE TABLE leads (
    id SERIAL PRIMARY KEY,
    full_name VARCHAR(100),
    email VARCHAR(100),
    company_name VARCHAR(100),
    company_size VARCHAR(20),
    service_type VARCHAR(50),
    budget_range VARCHAR(20),
    deadline VARCHAR(20),
    contact_channel VARCHAR(20),
    message_text TEXT,
    created_at TIMESTAMP,
    converted INTEGER
);

CREATE TABLE scores (
    id SERIAL PRIMARY KEY,
    lead_id INTEGER REFERENCES leads(id),
    score FLOAT,
    scored_at TIMESTAMP
);

\q
```

### 5. Update your database password
Open `database/db_connection.py` and set your password:
```python
conn = psycopg2.connect(
    host            = "localhost",
    database        = "tasmim_leads",
    user            = "postgres",
    password        = "Medi@@710", 
    port            = 5432,
    connect_timeout = 10
)
```

### 6. Set your Google Sheet ID
Open `data/sync_from_sheets.py` and set your Sheet ID:
```python
SHEET_ID = "10ZttGBxBB9S9tqda-CNKUmHbNzo6RFvmVeFKotMbjTI"   # ← from your Google Sheet URL
```

---

## 🚀 How to Run — Step by Step

Run these steps **in order** the first time.

### Step 1 — Start PostgreSQL
```bash
sudo service postgresql start
```

### Step 2 — Generate training data
Open `notebooks/generate_data.ipynb` in Jupyter and run all cells.
⚠️ This takes ~2-3 hours because of the T5 paraphrase model.

### Step 3 — Build the feature matrix
```bash
python features/build_features.py
```

### Step 4 — Train the model
```bash
python models/train_model.py
```
Saves: `best_model.pkl`, `scaler.pkl`, `feature_names.json`, `shap_importance.csv`

### Step 5 — Score all existing leads
```bash
python score_all_leads.py
```

### Step 6 — Start the API (keep this terminal open)
```bash
uvicorn api.main:app --reload
```
API runs at: `http://127.0.0.1:8000`
Interactive docs at: `http://127.0.0.1:8000/docs`

### Step 7 — Start the Dashboard (new terminal)
```bash
streamlit run dashboard/app.py
```
Dashboard runs at: `http://localhost:8501`

---

## 🎬 Demo Guide (for presentations)

### Before the demo
```bash
sudo service postgresql start
python score_all_leads.py
```

### During the demo — open 2 terminals

**Terminal 1 — API:**
```bash
uvicorn api.main:app --reload
```

**Terminal 2 — Dashboard:**
```bash
streamlit run dashboard/app.py
```

### What to show

**1. Dashboard home** (`http://localhost:8501`)
- Total leads, average score, high priority count
- Filter table by HIGH priority only

**2. Charts page** — sidebar → "1 charts"
- Score distribution
- Conversion by channel
- Leads per month

**3. API live test** (`http://127.0.0.1:8000/docs`)

High quality lead (should score HIGH):
```json
{
  "full_name": "Ahmed Benali",
  "email": "ahmed@techcorp.com",
  "company_name": "TechCorp",
  "company_size": "large",
  "service_type": "ecommerce",
  "budget_range": "enterprise",
  "deadline": "urgent",
  "contact_channel": "whatsapp",
  "message_text": "We need a full e-commerce platform with payment integration and inventory management. Budget is flexible for the right agency."
}
```

Low quality lead (should score LOW):
```json
{
  "full_name": "Random Person",
  "email": "random@gmail.com",
  "company_name": "unknown",
  "company_size": "solo",
  "service_type": "website",
  "budget_range": "low",
  "deadline": "flexible",
  "contact_channel": "social_media",
  "message_text": "hi website please how much"
}
```

**4. Google Form live demo**
- Fill the Google Form with a serious lead
- Run `python data/sync_from_sheets.py`
- Go to "2 score lead" page → select the new lead → click Score

**5. MLflow (bonus)**
```bash
mlflow ui
```
Open `http://127.0.0.1:5000` — shows all model runs and metrics.

---

## 🔄 Airflow Pipeline

Runs automatically every night at midnight:
```
sync_from_sheets → extract_leads → build_features → score_and_save
```

**Start Airflow (2 terminals):**
```bash
# Terminal 1
airflow scheduler

# Terminal 2
airflow webserver --port 8080
```

Open `http://localhost:8080`

**Default credentials:**
```
Username: admin
Password: admin123
```

**Reset credentials if forgotten:**
```bash
airflow users delete --username admin
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@tasmim.com \
    --password admin123
```

---

## 📬 Google Forms Integration

New leads from Google Form → Google Sheets → PostgreSQL → scored automatically.

**Manual sync:**
```bash
python data/sync_from_sheets.py
```

**Automatic sync:** runs as first task in Airflow pipeline every night.

**Requirements:**
- Google Sheet must be shared as "Anyone with the link can view"
- `SHEET_ID` must be set in `sync_from_sheets.py`

---

## 🔌 API Reference

### Score a single lead
```
POST http://127.0.0.1:8000/score
```

### Score multiple leads
```
POST http://127.0.0.1:8000/score/batch
```

### Health check
```
GET http://127.0.0.1:8000/health
```

### Interactive docs
```
GET http://127.0.0.1:8000/docs
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data generation | Faker, Hugging Face T5 |
| Database | PostgreSQL + psycopg2 |
| NLP | NLTK (VADER sentiment) |
| ML | scikit-learn, XGBoost |
| Explainability | SHAP |
| Confidence intervals | MAPIE (Conformal Prediction) |
| Experiment tracking | MLflow |
| API | FastAPI + uvicorn |
| Pipeline automation | Apache Airflow |
| Dashboard | Streamlit + Plotly |
| Google Forms sync | pandas |

---

## 📊 Model Performance

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.8205 | 0.6025 | 0.7348 | 0.6621 |
| **Random Forest** ✅ | **0.8256** | **0.7404** | **0.5833** | **0.6525** |
| XGBoost | 0.8244 | 0.6316 | 0.6364 | 0.6340 |

**Best model:** Random Forest — ROC-AUC: 0.8256

---

## 🔑 Score Interpretation

| Score | Priority | Action |
|---|---|---|
| 75 – 100 | 🔴 HIGH | Contact immediately |
| 45 – 74 | 🟡 MEDIUM | Follow up within 48 hours |
| 0 – 44 | 🟢 LOW | Send automated response |

---

## ⚠️ Important Notes

- Always start PostgreSQL before running any script: `sudo service postgresql start`
- Always start the API before opening the dashboard
- Run from WSL terminal — not Windows PowerShell
- Google Sheet must be public for the sync to work

---

## 👤 Author

- **Intern:** ELMEHDI MORTAJA
- **Company:** Tasmim Web — Casablanca
- **Period:** May – June 2026
- **Program:** 2nd year Data Engineering
