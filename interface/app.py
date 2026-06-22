import streamlit as st
import pandas as pd
import os
import sys
# To make importing from other folders possible
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_connection import get_connection


st.set_page_config(
    page_title='Tasmim Web - Lead Scoring',
    page_icon='interface/Logo-tasmim-web-creation-site-web.jpg',
    # layout='wide' uses the full browser width
    layout='wide',
    # It makes the sidebar opens by default
    initial_sidebar_state='expanded'
)

# This function allows streamlit to run a function just once
# and store the results in the memory
@st.cache_data(ttl=300)
def load_data():
    """
    Loads all scored leads from PostgreSQL.
    """

    try:
        conn = get_connection()
        query = """
        SELECT 
            l.id,
            l.full_name,
            l.email,
            l.company_name,
            l.company_size,
            l.service_type,
            l.budget_range,
            l.deadline,
            l.contact_channel,
            l.message_text,
            l.converted,
            l.created_at,
            s.score,
            s.scored_at
        FROM leads l
        INNER JOIN scores s ON l.id = s.lead_id
        ORDER BY s.score DESC
        """

        df = pd.read_sql(query, conn)
        conn.close()

        df['priority'] = pd.cut(
            df['score'],
            bins=[0,45,75,100],
            labels=['🟢 LOW', '🟡 MEDIUM', '🔴 HIGH']
        )
        # avoid to get a datetime as string
        df['created_at'] = pd.to_datetime(df['created_at'])

        return df
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return pd.DataFrame()

def show_metrics(df: pd.DataFrame):
    """
    Displays the 4 summary numbers at the the top of the dashboard.
    """

    st.subheader("📊 Overview")

    total_leads = len(df)
    avg_score = round(df['score'].mean(), 1)
    high_priority = len(df[df['score'] >= 75])
    conversion_rate = round(df['converted'].mean() * 100, 1)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label= 'Total Scored Leads',
            value= total_leads
        )
    
    with col2:
        st.metric(
            label='Average Score',
            value= avg_score
        )
    
    with col3:
        st.metric(
            label='High Priority Leads',
            value= high_priority,
            delta= f'{round(high_priority/total_leads * 100, 1)}% of all leads'
        )

    with col4:
        st.metric(
            label= 'Conversion Rate',
            value= f'{conversion_rate}%'
        )

    # Draw a line to separate
    st.divider()

def show_leads_table(df):
    """
    Displays a table combining:
    the lead + score informations.
    """

    st.subheader("📋 Scored Leads")

    with st.sidebar:
        st.header('🔍 Filters')
        
        all_priorities = df['priority'].unique().tolist()
        selected_priorities = st.multiselect(
            label= 'Priority',
            options= all_priorities,
            default= all_priorities
        )

        all_services = df['service_type'].unique().tolist()
        selected_services = st.multiselect(
            label= 'Service Type',
            options= all_services,
            default=  all_services
        )

        all_budgets = df['budget_range'].unique().tolist()
        selected_budgets = st.multiselect(
            label= 'Budget Range',
            options= all_budgets,
            default= all_budgets
        )

        all_companies = df['company_size'].unique().tolist()
        selected_companies = st.multiselect(
            label= 'Company Size',
            options= all_companies,
            default= all_companies
        )

        all_contacts = df['contact_channel'].unique().tolist()
        selected_contacts = st.multiselect(
            label= 'Contact Channel',
            options= all_contacts,
            default= all_contacts
        )

        st.divider()

        sort_by = st.selectbox(
            label= 'Sort by',
            options= ['Score (High to Low)', 'Score (Low to High)', 'Date (Newest first)']
        )

    filtered_df = df[
        df['priority'].isin(selected_priorities) &
        df['service_type'].isin(selected_services) &
        df['budget_range'].isin(selected_budgets) &
        df['company_size'].isin(selected_companies) &
        df['contact_channel'].isin(selected_contacts)
    ]

    if sort_by == "Score (High to Low)":
        filtered_df = filtered_df.sort_values('score', ascending=False)
    elif sort_by == "Score (Low to High)":
        filtered_df = filtered_df.sort_values('score', ascending=True)
    else:
        filtered_df = filtered_df.sort_values('created_at', ascending=False)

    st.info(f'Showing {len(filtered_df)} leads out of {len(df)} total')

    columns_to_show = [
        'full_name',
        'email',
        'company_name',
        'service_type',
        'budget_range',
        'deadline',
        'score',
        'priority',
        'contact_channel',
        'converted'
    ]

    st.dataframe(
        filtered_df[columns_to_show],
        use_container_width=True
    )
    st.divider()

def main():
    st.title('Tasmim Web - Lead Scoring Dashboard')
    st.write('Score and track all incoming leads in one place.')
    st.divider()

    df = load_data()

    if df.empty:
        st.warning('No scored leads found. Run the scoring pipeline first.')
        return 
    
    show_metrics(df)
    show_leads_table(df)

if __name__ == '__main__':
    main()