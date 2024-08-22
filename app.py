import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from wordcloud import WordCloud
import folium
from scipy import stats
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objs as go
from collections import Counter
from datetime import datetime, timedelta
import os
import pickle
from prophet import Prophet
import gdown


# file_id = '1WT2Mn3L0dktnU0pWvJ4NkTwRt0VImM9f'



# Google Drive URL
# url = f'https://drive.google.com/uc?export=download&id={file_id}'
# Destination path where the file will be saved
# output = dataset_path

# gdown.download(url, output, quiet=False)

# Load data

dataset_path = 'sampled_dataset.csv'
@st.cache_data
def load_data():
    
    df_cleaned = pd.read_csv(dataset_path)
    df_cleaned['job_posted_date'] = pd.to_datetime(df_cleaned['job_posted_date'])
    df_cleaned['date_only'] = df_cleaned['job_posted_date'].dt.date
    return df_cleaned

def process_skills(skills_string):
    skills_string = skills_string.replace('[', '').replace(']', '').replace('\'', '').replace(',', '')
    skills_list = skills_string.split()
    skills_list = ['power bi' if skill.lower() in ['power', 'bi'] else skill.lower() for skill in skills_list]
    skills_list = [skill for skill in skills_list if skill not in ['not', 'specified']]
    return skills_list

def recent_job_market(df_cleaned, eda_option):
    st.title("🌟 Recent Data Job Market")

    if eda_option == "📊 Data Jobs Posting":
        st.subheader("Job Posting In Different Data Fields")
        
        job_title_counts = df_cleaned['job_title_short'].value_counts().head(10)
        fig = px.bar(x=job_title_counts.index, y=job_title_counts.values, 
                     labels={'x': 'Job Title', 'y': 'Number of Jobs'}, 
                     title='Data Job Types by Number of Jobs',
                     color=job_title_counts.values,
                     color_continuous_scale='viridis')
        fig.update_layout(coloraxis_showscale=False, plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        text = ' '.join(df_cleaned['job_title_short'].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    elif eda_option == "📈 Job Trends":
        st.subheader("Job Trends Analysis")
        
        job_title_colors = {
            'Data Engineer': 'blue',
            'Data Analyst': 'green',
            'Data Scientist': 'red',
            'Machine Learning Engineer': 'purple',
            'Cloud Engineer': 'orange',
            'Business Analyst': 'pink',
            'Software Engineer': 'cyan'
        }

        fig, axes = plt.subplots(nrows=len(job_title_colors), ncols=1, sharex=True, figsize=(14, 24))
        axes = axes.flatten()

        for i, (job_title, color) in enumerate(job_title_colors.items()):
            filtered_df = df_cleaned[df_cleaned['job_title_short'] == job_title]
            jobs_per_day = filtered_df.groupby(filtered_df['job_posted_date'].dt.date).size()
            
            axes[i].plot(jobs_per_day.index, jobs_per_day.values, 
                         linestyle='-', color=color, label=f'{job_title} (Original)')
            
            x_values = np.arange(len(jobs_per_day))
            y_values = jobs_per_day.values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
            trend_line = slope * x_values + intercept
            
            axes[i].plot(jobs_per_day.index, trend_line, 
                         linestyle='--', color='black', label=f'{job_title} (Trend)')
            
            axes[i].set_title(job_title)
            axes[i].set_ylabel('Number of Jobs')
            axes[i].legend()

        plt.xlabel('Date')
        plt.tight_layout()
        st.pyplot(fig)

    elif eda_option == "🗺️ Job Locations":
        st.subheader("Job Locations Analysis")
        
        st.image("data_job_animation.gif", use_column_width=True)

    elif eda_option == "🛠️ Top Skills":
        st.subheader("Skills Analysis")
    
    # Step 1: Expand skills and associate with job titles
        skills_expanded = []
        for _, row in df_cleaned.dropna(subset=['job_skills']).iterrows():
            job_title = row['job_title_short']
            skills = row['job_skills'].replace('[', '').replace(']', '').replace('\'', '').replace(',', '').split()
            skills = [skill.lower() for skill in skills]
            skills = ['power bi' if skill in ['power', 'bi'] else skill for skill in skills]
            skills = [skill for skill in skills if skill not in ['not', 'specified']]
            for skill in skills:
                skills_expanded.append((skill, job_title))

    # Step 2: Convert to DataFrame
        skills_df = pd.DataFrame(skills_expanded, columns=['Skill', 'job_title_short'])

    # Step 3: Count frequencies of each skill per job title
        skills_counts = skills_df.groupby(['Skill', 'job_title_short']).size().unstack(fill_value=0)

    # Step 4: Sort by total counts and take top 20 skills
        top_skills_counts = skills_counts.sum(axis=1).sort_values(ascending=False).head(10)
        top_skills_df = skills_counts.loc[top_skills_counts.index]

    # Step 5: Define custom legend order
        custom_order = [
           'Data Engineer',
           'Data Analyst',
           'Data Scientist',
           'Machine Learning Engineer',
           'Cloud Engineer',
           'Business Analyst',
           'Software Engineer'
        ]
        # Filter custom_order to include only job titles present in the data
        available_job_titles = [title for title in custom_order if title in top_skills_df.columns]

    # Plotting with improvements
        fig, ax = plt.subplots(figsize=(16, 12))  # Increased figure size for better readability
    
        if not available_job_titles:
           st.warning("No matching job titles found in the data.")
           return

        # fig, ax = plt.subplots(figsize=(16, 12))  # Increased figure size for better readability
        top_skills_df[custom_order].plot(kind='barh', stacked=True, ax=ax, cmap='Spectral')

        # Reverse the y-axis to show highest counts at the top
        ax.invert_yaxis()

        # Improve x-axis labels
        ax.set_xlabel('Count', fontsize=16)
        ax.tick_params(axis='x', labelsize=14)

        # Improve y-axis labels
        ax.set_ylabel('Skill', fontsize=16)
        ax.tick_params(axis='y', labelsize=16)

        # Add title with improved font size
        ax.set_title('Top 10 Job Skills by Job Title', fontsize=18, fontweight='bold')

        # Improve legend
        ax.legend(title='Job Title', title_fontsize=14, fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add value labels to the end of each bar
        for i, skill in enumerate(top_skills_df.index):
            total = top_skills_df.loc[skill].sum()
            ax.text(total, i, f' {total}', va='center', ha='left', fontsize=10)

        # Adjust layout to prevent cutting off labels
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)

    elif eda_option == "💼 Top Countries & Companies":
        st.subheader("World's Top Countries and Companies posting Data Jobs")
        
        company_job_counts = df_cleaned['company_name'].value_counts().head(10)
        company_countries = df_cleaned.groupby('company_name')['job_country'].agg(lambda x: x.mode()[0])
        country_job_counts = df_cleaned['job_country'].value_counts().head(10)

        fig = make_subplots(rows=1, cols=2, subplot_titles=('Top 10 Countries by Job Postings', 'Top 10 Companies by Number of Jobs'))

        fig.add_trace(
            go.Bar(x=country_job_counts.index, y=country_job_counts.values, 
                   marker_color=px.colors.sequential.Plasma),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(x=company_job_counts.index, y=company_job_counts.values, 
                   marker_color=px.colors.sequential.Plasma,
                   text=[company_countries[company] for company in company_job_counts.index],
                   textposition='inside'),
            row=1, col=2
        )

        fig.update_layout(height=600, showlegend=False, title_text="Top Countries and Companies")
        fig.update_xaxes(tickangle=45)

        st.plotly_chart(fig, use_container_width=True)

    elif eda_option == "💰 Salaries":
        st.subheader("Salary Analysis")
        
        grouped_df = df_cleaned.groupby('job_title_short').agg({'num_jobs': 'sum', 'salary_year_avg': 'mean'}).reset_index()
        
        fig = px.scatter(grouped_df, x='salary_year_avg', y='num_jobs', size='num_jobs', color='job_title_short',
                         hover_name='job_title_short', size_max=60,
                         labels={'salary_year_avg': 'Average Annual Salary', 'num_jobs': 'Total Number of Jobs'},
                         title='Salary vs. Total Job Frequency by Job Title')
        fig.update_layout(legend_title='Job Title', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

def future_job_trends(df_cleaned):
    st.title("🔮 Future Job Trends")

    job_types = sorted(df_cleaned['job_title_short'].unique())
    cols = st.columns(4)
    selected_job_type = None
    for i, job_type in enumerate(job_types):
        if cols[i % 4].button(f"🔍 {job_type}", key=f"job_type_{i}"):
            selected_job_type = job_type

    st.markdown("---")

    if selected_job_type:
        st.subheader(f"📊 Trends for {selected_job_type}")
        forecast_path = f"forecast_plots/{selected_job_type}_forecast.png"
        if os.path.exists(forecast_path):
            st.image(forecast_path, caption=f"Forecast for {selected_job_type}", use_column_width=True)
        else:
            st.warning(f"No forecast plot available for {selected_job_type}")

        # Top 10 countries with highest job postings
        top_countries = df_cleaned[df_cleaned['job_title_short'] == selected_job_type]['job_country'].value_counts().head(10)
        
        fig = px.bar(x=top_countries.index, y=top_countries.values,
                     labels={'x': 'Country', 'y': 'Number of Jobs'},
                     title=f'Top 10 Countries for {selected_job_type}',
                     color=top_countries.values,
                     color_continuous_scale='Viridis')
        fig.update_layout(
            coloraxis_showscale=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=14, color='#FFFFFF'),
            title=dict(font=dict(size=24, color='#FF6B6B')),
            xaxis=dict(title=dict(font=dict(size=18, color='#4ECDC4'))),
            yaxis=dict(title=dict(font=dict(size=18, color='#4ECDC4')))
        )
        st.plotly_chart(fig, use_container_width=True)

        # Average salary and top job portals
        job_data = df_cleaned[df_cleaned['job_title_short'] == selected_job_type]
        avg_salary = job_data['salary_year_avg'].mean()
        top_portals = job_data['job_via'].value_counts().head(3)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("💰 Average Salary", f"${avg_salary:,.2f}")
        with col2:
            st.markdown("### 🏆 Top 3 Job Portals:")
            for portal, count in top_portals.items():
                st.markdown(f"- **{portal}**: {count}")

@st.cache_resource
def load_model(job_type):
    model_path = f'model_files/{job_type}_job_prediction_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    return None

def preparing_for_opportunities(df_cleaned):
    st.title("🚀 Preparing for Tomorrow's Opportunities")

    all_skills = ' '.join(df_cleaned['job_skills'].dropna())
    processed_skills = process_skills(all_skills)
    skill_counts = Counter(processed_skills)
    top_skills = [skill for skill, _ in skill_counts.most_common(50)]
    selected_skills = st.multiselect("🛠️ Select skills:", top_skills)

    locations = ["All"] + sorted(df_cleaned['job_country'].unique().tolist())
    selected_location = st.selectbox("🌍 Select location:", locations)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("📅 Select start date for prediction", 
                                   min_value=datetime.now().date(),
                                   max_value=datetime(2024, 12, 31).date(),
                                   value=datetime.now().date())
    with col2:
        end_date = st.date_input("📅 Select end date for prediction", 
                                 min_value=start_date,
                                 max_value=datetime(2024, 12, 31).date(),
                                 value=min(start_date + timedelta(days=30), datetime(2024, 12, 31).date()))

    st.markdown("---")

    if start_date > datetime(2024, 12, 31).date() or end_date > datetime(2024, 12, 31).date():
        st.error("⚠️ Please select a date range within the year 2024.")
    elif start_date > end_date:
        st.error("⚠️ End date must be after start date.")
    elif selected_skills:
        df_filtered = df_cleaned[
            (df_cleaned['job_skills'].apply(lambda x: any(skill in x for skill in selected_skills))) &
            (df_cleaned['job_country'] == selected_location if selected_location != "All" else True)
        ]
        
        job_types = df_filtered['job_title_short'].value_counts()
        job_types_proportion = job_types / job_types.sum()

        fig = px.bar(x=job_types_proportion.index, y=job_types_proportion.values,
                     labels={'x': 'Job Title', 'y': 'Skill relevance proportion in Data job roles'},
                     title='Relevant Job Types',
                     color=job_types_proportion.values,
                     color_continuous_scale='Viridis')
        fig.update_layout(
            coloraxis_showscale=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=14, color='#FFFFFF'),
            title=dict(font=dict(size=24, color='#FF6B6B')),
            xaxis=dict(title=dict(font=dict(size=18, color='#4ECDC4'))),
            yaxis=dict(
                title=dict(text='Skill relevance proportion in Data job roles', font=dict(size=18, color='#4ECDC4')),
                showticklabels=False
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        selected_job = st.selectbox("🔍 Select a job type for details:", job_types.index.tolist())
        
        if selected_job:
            job_data = df_filtered[df_filtered['job_title_short'] == selected_job]
            
            # Load and use the prediction model
            model = load_model(selected_job)
            if model:
                future_dates = pd.date_range(start=start_date, end=end_date)
                future = pd.DataFrame({'ds': future_dates})
                
                # Make prediction
                forecast = model.predict(future)
                
                # Adjust prediction based on selected location
                if selected_location != "All":
                    # Calculate the proportion of jobs in the selected country
                    country_proportion = len(df_cleaned[(df_cleaned['job_title_short'] == selected_job) & 
                                                        (df_cleaned['job_country'] == selected_location)]) / \
                                         len(df_cleaned[df_cleaned['job_title_short'] == selected_job])
                    
                    # Adjust the forecast
                    forecast['yhat'] *= country_proportion
                    forecast['yhat_lower'] *= country_proportion
                    forecast['yhat_upper'] *= country_proportion
                
                # Calculate total predicted jobs
                total_jobs = int(forecast['yhat'].sum())
                
                # Calculate average salary based on selected location
                if selected_location == "All":
                    avg_salary = job_data['salary_year_avg'].mean()
                    salary_label = "Global Average Salary"
                else:
                    avg_salary = job_data[job_data['job_country'] == selected_location]['salary_year_avg'].mean()
                    salary_label = f"Average Salary in {selected_location}"
                
                top_providers = job_data['job_via'].value_counts().nlargest(3).index.tolist()

                st.markdown(f"### 💼 {selected_job}")
                col1, col2, col3 = st.columns(3)
                col1.metric("📊 Predicted Total Jobs", f"{total_jobs:,}")
                col2.metric(f"💰 {salary_label}", f"${avg_salary:,.2f}")
                col3.markdown("### 🏆 Top 3 Job Portals:")
                col3.write(", ".join(top_providers))

                # Display the prediction chart
                fig = px.line(forecast, x='ds', y='yhat', 
                              title=f'Predicted Job Postings for {selected_job} {"in " + selected_location if selected_location != "All" else "Globally"}')
                fig.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], 
                                fill=None, mode='lines', line_color='rgba(0,100,80,0.2)', name='Lower Bound')
                fig.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], 
                                fill='tonexty', mode='lines', line_color='rgba(0,100,80,0.2)', name='Upper Bound')
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=14, color='#FFFFFF'),
                    title=dict(font=dict(size=24, color='#FF6B6B')),
                    xaxis=dict(title=dict(text='Date', font=dict(size=18, color='#4ECDC4'))),
                    yaxis=dict(title=dict(text='Predicted Job Postings', font=dict(size=18, color='#4ECDC4')))
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No prediction model available for {selected_job}")

        
        st.markdown("---")

        # Work from home percentage
        wfh_counts = df_filtered['job_work_from_home'].value_counts(normalize=True) * 100
        st.subheader("🏠 Work Environment: Remote or Onsite?")
        fig_wfh = px.bar(x=['On-site', 'Remote'], y=wfh_counts.values, 
                         text=[f'{v:.1f}%' for v in wfh_counts.values],
                         title="Work From Home vs On-site")
        fig_wfh.update_traces(textposition='outside', marker_color=['#FF6B6B', '#4ECDC4'])
        fig_wfh.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=14, color='#FFFFFF'),
            title=dict(font=dict(size=24, color='#FF6B6B')),
            xaxis=dict(title=dict(font=dict(size=18, color='#4ECDC4'))),
            yaxis=dict(title=dict(text='Percentage', font=dict(size=18, color='#4ECDC4')))
        )
        st.plotly_chart(fig_wfh, use_container_width=True)

        # Job schedule type percentage
        schedule_counts = df_filtered['job_schedule_type'].value_counts(normalize=True) * 100
        st.subheader("⏰  Possible Job Schedule")
        fig_schedule = px.bar(x=schedule_counts.index, y=schedule_counts.values, 
                              text=[f'{v:.1f}%' for v in schedule_counts.values],
                              title="Job Schedule Types")
        fig_schedule.update_traces(textposition='outside', marker_color='#45B7D1')
        fig_schedule.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=14, color='#FFFFFF'),
            title=dict(font=dict(size=24, color='#FF6B6B')),
            xaxis=dict(title=dict(font=dict(size=18, color='#4ECDC4'))),
            yaxis=dict(title=dict(text='Percentage', font=dict(size=18, color='#4ECDC4')))
        )
        st.plotly_chart(fig_schedule, use_container_width=True)

    else:
        st.warning("⚠️ Please select at least one skill to see job recommendations.")

def about_page():
    st.title("ℹ️ About This Project")
    
    st.markdown("""
    ## 🎯 Project Aim
    This project aims to assist job seekers in the data field by providing recent job trends and forecasts.

    ## 📊 Data Source
    The raw data, comprising over 780,000 rows of global job listings from 2023, was sourced from Hugging Face.

    ## 🔍 Data Processing
    The data underwent thorough cleaning and analysis to extract valuable insights about the current job market in the data field.

    ## 📈 Forecasting
    Time series prediction was performed using the PROPHET model to forecast future job trends.

    ## 🖥️ Application
    This interactive, dynamic app utilizes the processed data and predictive models to provide users with valuable job market insights and forecasts.
    """)

def main():
    st.set_page_config(layout="wide", page_title="Emerging Data Job Opportunities", page_icon="📊")
    
    # Custom CSS
    st.markdown("""
    <style>
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    
    .main-title {
        font-size: 36px;
        font-weight: bold;
        color: #FFD700;
        text-align: center;
        padding: 10px 0;
        background: #2E1A47;
        border-bottom: 3px solid #FFD700;
        margin-bottom: 20px;
    }
    .sidebar .sidebar-content {
        background-color: #2C3E50;
    }
    .sidebar-option {
        font-size: 24px;
        font-weight: bold;
        color: #8E44AD;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .sidebar-suboption {
        font-size: 18px;
        color: #4ECDC4;
        margin-left: 20px;
    }
    .stButton>button {
        background-color: #3A506B;
        color: #FFFFFF;
        font-weight: bold;
        border: 2px solid #4ECDC4;
        border-radius: 5px;
        padding: 10px 15px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #4ECDC4;
        color: #2E1A47;
    }
    .main-page-name {
        font-size: 24px;
        font-weight: bold;
        color: #4A0E4E;
        margin-bottom: 10px;
    }
    .creator-info {
        font-size: 18px;
        color: #1A5F59;
        text-align: left;
        margin-top: 10px;
        margin-bottom: 15px;
        padding: 10px;
        background-color: rgba(78, 205, 196, 0.1);
        border-radius: 5px;
        border-left: 4px solid #4ECDC4;
    }
    .creator-info a {
        color: #FFD700;
        text-decoration: none;
        font-weight: bold;
    }
    .creator-info a:hover {
        text-decoration: underline;
    }
    .prediction-note {
        position: fixed;
        top: 60px;
        right: 10px;
        background-color: #E6F3FF;
        color: #2C3E50;
        padding: 10px;
        border-radius: 5px;
        font-size: 14px;
        border-left: 4px solid #3498DB;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        z-index: 1000;
        max-width: 300px;
        display: flex;
        align-items: center;
    }
    .prediction-note i {
        font-size: 20px;
        color: #3498DB;
        margin-right: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main title
    st.markdown('<p class="main-title">Emerging Data Job Opportunities</p>', unsafe_allow_html=True)
    
    df_cleaned = load_data()
    
    # Add the cropped GIF to the top of the sidebar
    st.sidebar.markdown('<div class="sidebar-image">', unsafe_allow_html=True)
    st.sidebar.image("data_job_animation.gif", use_column_width=True)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Add creator info below the GIF
    st.sidebar.markdown("""
    <div class="creator-info">
        <span style="font-size: 24px;">🧑‍💻</span> Created by Reetu Sharma<br>
        <span style="font-size: 24px;">🔗</span> <a href="https://www.linkedin.com/in/dr-reetu-sharma-73991348" target="_blank">LinkedIn Profile</a>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.markdown('<p class="sidebar-option">Main Pages</p>', unsafe_allow_html=True)
    
    # Create custom radio buttons for main pages
    main_pages = ["Recent Data Job Market", "Upcoming Possibilities Of Jobs In Data", "About"]
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = main_pages[0]

    for page in main_pages:
        if st.sidebar.button(page, key=f"btn_{page}", help=f"Navigate to {page}"):
            st.session_state.current_page = page

    st.sidebar.markdown(f'<p class="main-page-name">{st.session_state.current_page}</p>', unsafe_allow_html=True)
    
    
    if st.session_state.current_page == "Recent Data Job Market":
        st.sidebar.markdown('<p class="sidebar-suboption">Choose EDA Option:</p>', unsafe_allow_html=True)
        eda_option = st.sidebar.radio("", [
            "📊 Data Jobs Posting",
            "📈 Job Trends",
            "🗺️ Job Locations",
            "💼 Top Countries & Companies",
            "💰 Salaries",
            "🛠️ Top Skills"
        ], label_visibility="collapsed")
        recent_job_market(df_cleaned, eda_option)
    elif st.session_state.current_page == "Upcoming Possibilities Of Jobs In Data":
        st.sidebar.markdown('<p class="sidebar-suboption">Choose Option:</p>', unsafe_allow_html=True)
        sub_page = st.sidebar.radio("", ["Future Job Trends", "Preparing for Tomorrow's Opportunities"], label_visibility="collapsed")
        if sub_page == "Future Job Trends":
            future_job_trends(df_cleaned)
        else:
            st.markdown("""
            <div class="prediction-note">
                <i class="fas fa-robot"></i>
                The predictions are based on AI model and may differ from actual results.
            </div>
            """, unsafe_allow_html=True)
            preparing_for_opportunities(df_cleaned)
    else:  # About page
        about_page()

if __name__ == "__main__":
    main()