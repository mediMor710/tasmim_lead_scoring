import streamlit as st
import pandas as pd
import requests
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_connection import get_connection

st.set_page_config(
    page_title = 'Score a Lead - Tasmim Web',
    page_icon = '🎯',
    layout = 'wide'
)

@st.cache_data(ttl=60)
def load_unscored_leads():
    """
    Loads all leads that don't have a score yet.
    """
    try:
        conn = get_connection()
        query = """
            SELECT l.*
            FROM leads l
            LEFT JOIN scores s
            ON l.id = s.lead_id
            WHERE s.id IS NULL
            ORDER BY l.created_at DESC
        """
        df = pd.read_sql(query, conn)
        conn.close()

        return df
    except Exception as e:
        st.error(f'Database error: {e}')
        return pd.DataFrame()
    
def main():

    st.title('🎯 Score a Lead')
    st.write('Pick an unscored lead from the dropdown and score it instantly.')
    st.divider()

    df = load_unscored_leads()

    if df.empty:
        st.success('All leads have been scored already!')
        st.write('New leads will apear here after they submit the Google Form.')
        return
    
    st.info(f'Found **{len(df)}** unscored leads.')

    df['label'] = (
        df['full_name'] + ' - ' + 
        df['company_name'] + ' (' + 
        df['service_type'] + ')'
    )

    selected_label = st.selectbox(
        label = 'Select a lead to score',
        options = df['label'].tolist()
    )

    selected_lead = df[df['label'] == selected_label].iloc[0]

    st.subheader('Lead Details')

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Name:**    {selected_lead['full_name']}")
        st.write(f"**Email:**   {selected_lead['email']}")
        st.write(f"**Company:** {selected_lead['company_name']}")
        st.write(f"**Size:**    {selected_lead['company_size']}")
        st.write(f"**Channel:** {selected_lead['contact_channel']}")

    with col2:
        st.write(f"**Service:**  {selected_lead['service_type']}")
        st.write(f"**Budget:**   {selected_lead['budget_range']}")
        st.write(f"**Deadline:** {selected_lead['deadline']}")

    with st.expander('📩 View Message'):
        st.write(selected_lead['message_text'])

    st.divider()

    if st.button('🎯 Score this lead', type='primary'):
        with st.spinner('Scoring...'):
            try:
                lead_data = {
                    "full_name":       str(selected_lead['full_name']),
                    "email":           str(selected_lead['email']),
                    "company_name":    str(selected_lead['company_name']),
                    "company_size":    str(selected_lead['company_size']),
                    "service_type":    str(selected_lead['service_type']),
                    "budget_range":    str(selected_lead['budget_range']),
                    "deadline":        str(selected_lead['deadline']),
                    "contact_channel": str(selected_lead['contact_channel']),
                    "message_text":    str(selected_lead['message_text'])
                }

                response = requests.post(
                    'http://127.0.0.1:8000/score',
                    json = lead_data,
                    timeout = 10
                )

                if response.status_code == 200:
                    result = response.json()
                    score = result['score']
                    priority = result['priority']

                    st.divider()
                    st.subheader('Result')

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric('Score', f'{score} / 100')
                    with col2:
                        st.metric('Priority', priority)
                    with col3:
                        st.metric(
                            'Conversion Probability',
                            f'{round(result['conversion_probability'] * 100, 1)}%'
                        )

                    if priority == 'HIGH':
                        st.success(f'✅ {result['recommendation']}')
                    elif priority == 'MEDUIM':
                        st.warning(f'⚠️ {result['recommendation']}')
                    else:
                        st.info(f'{result['recommendation']}')

                    with st.expander('📊 See top scoring factors'):
                        for feature, value in result['top_factors'].items():
                            st.write(f'• **{feature}**: {round(float(value), 4)}')

                    st.cache_data.clear()

                else:
                    st.error(f'API error {response.status_code}: {response.text}')
                
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot reach the API. Make sure it's running: `uvicorn API.main:app --reload`")
            except Exception as e:
                st.error(f'Something went wrong: {e}')

        
if __name__ == '__main__':
    main()