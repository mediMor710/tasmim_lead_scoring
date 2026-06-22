import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gspread
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import pandas as pd
from datetime import datetime
import json
from database.db_connection import get_connection

SHEET_ID = '10ZttGBxBB9S9tqda-CNKUmHbNzo6RFvmVeFKotMbjTI'
TRACKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'last_synced_row.json')

print('Sync_from_sheets.py loaded!')

def load_sheet_data():
    """
    Reads all row from the google sheet.
    """

    try:
        url = (
            f"https://docs.google.com/spreadsheets/d/{SHEET_ID}"
            f"/gviz/tq?tqx=out:csv&sheet=Form+Responses+1"
        )
        df = pd.read_csv(url)
        print(f'Sheet loaded - {len(df)} total rows found')

        return df
    
    except Exception as e:
        print(f'Could not load sheet: {e}')
        return pd.DataFrame()
    
def get_last_synced_row():
    """
    Reads the last synced row number from the tracker file.
    returns 0 if the file doesn't exist yet (First time running)
    """

    if os.path.exists(TRACKER_PATH):
        try:
            with open(TRACKER_PATH, 'r') as f:
                data = json.load(f)
            return data.get('last_row')
        
        except: 
            return 0
    
    else:
        return 0
    
def save_last_synced_row(row_number):
    """
    saves the last synced row number to the tracker file.
    """

    with open(TRACKER_PATH, 'w') as f:
        json.dump({'last_row': row_number}, f)

    print(f'Tracker updated - last synced row: {row_number}')

def convert_row_to_lead(row):
    """
    converts one Google sheet row into a lead dictionary
    that matches the PostgreSQL leads table structure.
    """

    try:
        # Defines 'Unkown' value to empty rows to avoid crashing
        full_name = str(row.get('Full Name', '')).strip()
        company_name = str(row.get('Company Name', '')).strip()

        if not full_name or not company_name:
            print('Skipping empty row')
            return None
        
        lead = {}

        
        lead['full_name'] = full_name
        lead['email'] = str(row.get('Email', '')).strip()
        lead['company_name'] = str(row.get('Company Name', 'Unkown')).strip()
        lead['company_size'] = str(row.get('Company Size', '')).strip().lower()
        lead['service_type'] = str(row.get('Service Type', '')).strip().lower()
        lead['budget_range'] = str(row.get('Budget Range', '')).strip().lower()
        lead['deadline'] = str(row.get('Deadline', '')).strip().lower()
        lead['contact_channel'] = str(row.get('Contact Channel', '')).strip().lower()
        lead['message_text'] = str(row.get('Message', '')).lower()
        lead['converted'] = 0
        lead['created_at'] = datetime.now()

        return lead
    
    except Exception as e:
        print(f'Could not convert row: {e}')
        return None
    
def sync_new_leads():
    """
    The main function that combine all the previous functions.
    Load rows -> Find new ones -> Convert them to a lead dict
    -> Insert into PostgreSQL -> Update the tracker file
    """

    print("#"*30)
    print('SYNC - Google Sheets -> PostgreSQL')
    print('#'*30)

    df = load_sheet_data()
    if df.empty:
        print('No data in sheet yet.')
        return 0
    
    last_row = get_last_synced_row()
    # Getting everything after last row
    new_rows = df.iloc[last_row:]

    print(f'Last synced row: {last_row}')
    print(f'New rows found: {len(new_rows)}')

    if len(new_rows) == 0:
        print('Nothing new to sync')
        return 0
    
    leads_to_insert = []
    
    for _, row in new_rows.iterrows():
        lead = convert_row_to_lead(row)
        if lead is not None:
            leads_to_insert.append(lead)

    print(f'Valid leads to insert: {len(leads_to_insert)}')

    if len(leads_to_insert) == 0:
        print('No valid leads found in new rows.')
        return 0
    
    conn = get_connection()
    cursor = conn.cursor()

    insert_query = """
        INSERT INTO leads(
        full_name, email, company_name, company_size,
            service_type, budget_range, deadline,
            contact_channel, message_text, created_at, converted
        ) VALUES (
            %(full_name)s, %(email)s, %(company_name)s, %(company_size)s,
            %(service_type)s, %(budget_range)s, %(deadline)s,
            %(contact_channel)s, %(message_text)s, %(created_at)s, %(converted)s
            )
    """

    for lead in leads_to_insert:
        cursor.execute(insert_query, lead)

    conn.commit()
    cursor.close()
    conn.close()

    print(f"Inserted {len(leads_to_insert)} new leads into PostgreSQL")

    # Update the last row
    save_last_synced_row(last_row + len(new_rows))

    return len(leads_to_insert)

if __name__ == '__main__':
    total = sync_new_leads()
    print(f'Done - {total} new leads synced.')