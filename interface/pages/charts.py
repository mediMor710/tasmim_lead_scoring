import streamlit as st
import pandas as pd
import plotly.express as px
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_connection import get_connection

st.set_page_config(
    page_title='Charts - Tasmim Web',
    page_icon = '📈',
    layout='wide'
)

@st.cache_data(ttl=300)
def load_data():
    try:
        conn = get_connection()
        query = """
            SELECT l.*, s.score, s.scored_at
            FROM leads l
            INNER JOIN scores s 
            ON l.id = s.lead_id
        """
        df = pd.read_sql(query, conn)
        conn.close()
        df['created_at'] = pd.to_datetime(df['created_at'])
        return df
    except Exception as e:
        st.error(f'Database error: {e}')
        return pd.DataFrame()
    
def main():

    st.title('📈 Charts & Analytics')
    st.write('Visual breakdown of lead scoring results.')
    st.divider()

    df = load_data()

    if df.empty:
        st.warning('No data found.')
        return
    
    st.subheader('Score Distribution')
    st.write('How are scores spread across all leads?')

    fig1 = px.histogram(
        df,
        x = 'score',
        nbins = 20,
        labels = {'score': 'Score', 'count': 'Number of leads'},
        color_discrete_sequence = ['#636EFA']
    )
    fig1.update_layout(
        showlegend = False,
        plot_bgcolor = 'rgba(0,0,0,0)',
        paper_bgcolor = 'rgba(0,0,0,0)',
        height = 350,
        margin = dict(l=20, r=20, t=20, b=20)
    )
    
    st.plotly_chart(fig1, use_container_width=True)

    st.divider()

    st.subheader('Conversion Rate by Channel')
    st.write('Which contact channel converts best?')

    channel_stats = df.groupby('contact_channel')['converted'].mean().reset_index()
    channel_stats.columns = ['channel', 'conversion_rate']
    channel_stats['conversion_rate'] = (channel_stats['conversion_rate'] * 100).round(1)
    channel_stats = channel_stats.sort_values('conversion_rate',ascending=False)

    fig2 = px.bar(
        channel_stats,
        x = 'channel',
        y = 'conversion_rate',
        labels = {'channel': 'Channel', 'conversion_rate': 'Conversion Rate (%)'},
        color = 'conversion_rate',
        color_continuous_scale = 'Blues',
        # Showing the exact value of conversion rate on each bar
        text = 'conversion_rate'
    )
    # adding '%' to the conversion value and moves it on top of the bar
    fig2.update_traces(texttemplate='%{text}%', textposition='outside')
    fig2.update_layout(
        showlegend = False,
        plot_bgcolor = 'rgba(0,0,0,0)',
        paper_bgcolor = 'rgba(0,0,0,0)',
        height = 400,
        margin = dict(l=20, r=20, t=20, b=20)
    )

    st.plotly_chart(fig2, use_container_width=True)
    st.divider()

    st.subheader('Conversion Rate by Company Size')
    st.write('Which company size converts best?')

    company_stats = df.groupby('company_size')['converted'].mean().reset_index()
    company_stats.columns = ['company_size', 'conversion_rate']
    company_stats['conversion_rate'] = (company_stats['conversion_rate'] * 100).round(1)
    company_stats = company_stats.sort_values('conversion_rate', ascending=False)

    fig3 = px.bar(
        company_stats,
        x = 'company_size',
        y = 'conversion_rate',
        labels = {'company_size': 'Company Size', 'conversion_rate': 'Conversion Rate (%)'},
        color='conversion_rate',
        color_continuous_scale = 'Greens',
        text = 'conversion_rate'
    )
    fig3.update_traces(texttemplate='%{text}%', textposition='outside')
    fig3.update_layout(
        showlegend = False,
        plot_bgcolor = 'rgba(0,0,0,0)',
        paper_bgcolor = 'rgba(0,0,0,0)',
        height = 400,
        margin = dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.divider()

    st.subheader('Leads Per Month')
    st.write('How many leads arrived each month?')

    # Convert the result to string so plotly can use it
    df['month'] = df['created_at'].dt.to_period('M').astype(str)
    monthly = df.groupby('month').size().reset_index()
    monthly.columns = ['month', 'lead_count']
    monthly = monthly.sort_values('month')

    fig4 = px.line(
        monthly,
        x = 'month',
        y = 'lead_count',
        labels = {'month': 'Month', 'lead_count': 'Number of leads'},
        markers = True,
        color_discrete_sequence=['#00CC96']
    )
    fig4.update_layout(
        plot_bgcolor = 'rgba(0,0,0,0)',
        paper_bgcolor = 'rgba(0,0,0,0)',
        height = 350,
        margin = dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig4,use_container_width=True)

if __name__ == '__main__':
    main()

