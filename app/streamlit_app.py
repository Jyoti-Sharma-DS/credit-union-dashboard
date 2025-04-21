import os
import sys
_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(_path)
import pandas as pd
import numpy as np
import random
import streamlit as st
import warnings
from scripts.kpi_analysis import compute_agent_kpis
from scripts.clustering import run_kmeans_clustering, asses_clusters_quality
from scripts.stats_tests import run_anova #, run_tukey_test
from scripts.config import task_type_mapping, outcome_mapping, clusters_mapping, cluster_dict,cluster_rec
from scripts.data_prep import load_and_prepare_data, map_task_categories, map_outcome_categories
from scripts.utils import calculate_outlier_stats , load_data
from scripts.visuals import (
    line_moving_average_plot_plotly,
    gen_count_plt_plotly,
    gen_percentage_plt_plotly, 
    plot_handle_time_vs_success_rate_plotly,
    plot_cluster_scatter_plotly, 
    plot_handle_time_boxplot,
    plot_categorical_associations_dython,
    plot_agent_task_category_distribution,
    plot_agent_task_vs_outcome_distribution,
    plot_agent_productivity,
)


# st.cache_data.clear()     # Clears data cache
# st.cache_resource.clear() # Clears resource cache (e.g., models, connections)
# set the global seed
# Set the random seed
random_seed=1212
# Set the random seed for python environment 
os.environ['PYTHONHASHSEED']=str(random_seed)
# Set numpy random seed
np.random.seed(random_seed)
# Set the random seed value
random.seed(random_seed)

# Filter out the warnings
warnings.filterwarnings('ignore')

# ------------------- Streamlit Page Setup ------------------- #
st.set_page_config(page_title="Agent Performance Dashboard", layout="wide")
st.title("📞 Credit Union Agent Performance Dashboard")

# ------------------- Load Data ------------------- #
# Load your datase
data_path = "data/bq-results-20230523-190701-1684868834975.csv"
df = load_data(data_path)

# Check if data was loaded successfully
if df is None:
    st.stop()  # Gracefully stop the app if data isn't usable
else:
    st.success(f"✅ Data loaded successfully — {df.shape[0]:,} records and {df.shape[1]} columns")

try:
    #convert the data types
    df['AGENT_ID'] = df['AGENT_ID'].astype('string')
    df['TASK_TYPE'] = df['TASK_TYPE'].astype('string')
    df['OUTCOME'] = df['OUTCOME'].astype('string')
    df['REQUEST_TYPE'] = df['REQUEST_TYPE'].astype('string') 
    df['TASK_DATE'] = pd.to_datetime(df['TASK_DATE'], errors='coerce', infer_datetime_format=True)
    df['DATE'] =  df['TASK_DATE'].dt.strftime('%Y-%m-%d')
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce', infer_datetime_format=True)


    df = map_task_categories(df, task_type_mapping)  
    df = map_outcome_categories(df, outcome_mapping)
    
    st.markdown("""
    ## 📊 **Executive Summary: Agent Performance Analysis**
     
    This analysis evaluates employee performance using task-level data from a 85 days period, covering over 261,369 records completed by 15 agents. The objective is to uncover performance patterns, operational trends, and data-driven recommendations for improving task efficiency and agent effectiveness.
    
    **Key Highlights:**
    
    ***Workload Distribution:***
    Identified disparities in task volume across agents and task categories, with "Research" being the most common task type.
    
    ***Performance Metrics:***
    Computed agent-level KPIs such as:
    
    - Average tasks per day
    - Average handle time
    - Task success rate
    - Issue frequency
    
    ***Trend Analysis:***
    Time-series analysis revealed daily workload fluctuations with a stable long-term trend around ~3000 tasks/day.
    
    ***Correlation Insights:***
    Clear relationships between:
    
    - Agents and task specialization
    - Task types and successful outcomes
    
    ***Clustering & Segmentation:***
    Applied KMeans clustering to segment agents into:
    
    - High performers
    - Balanced performers
    - Agents needing support
    
    **Outcome:**
    This analysis enables data-backed decisions around task allocation, performance coaching, and process improvement to optimize both individual and team efficiency.
    
    """ )

    
    # ------------------- Initial Visualizations (EDA) ------------------- #
    st.subheader(" 🧾 1. Overview & Exploratory Analysis: Understanding the Data ")
    
    Total_task = len(df)
    Total_Agents = df['AGENT_ID'].nunique()
    Total_No_Of_Days =  df['DATE'].nunique()
    Start_Date =  df['DATE'].min().strftime('%Y-%m-%d')
    End_Date = df['DATE'].max().strftime('%Y-%m-%d')
    
    #with st.expander("🔽 Click to view summary"):
    st.write(
        "We are analyzing **{}** records for **{}** different agents. "
        "The analysis covers a period from **{}** to **{}**, spanning **{} days**.".format(
            Total_task,
            Total_Agents,
            Start_Date ,
            End_Date,
            Total_No_Of_Days
        )
    ) 
    st.markdown("---") 
    ## ---------------- Generate Task (#) for all the Agents
    st.markdown("This section explores features of the dataset and analyses them for patterns if any.")
    with st.expander("🔽 Click to view trend analysis & plots"):

        st.markdown("#####  Task(#) By Agent Id  (% Distribution)")
        st.markdown("In this section we will explore Task(#) distribution by agent id")
        gen_percentage_plt_plotly(df, 'AGENT_ID' )
        st.markdown("""
##### 💡 Insights from Task Distribution:
    
- **Agent A** appears to have the highest task volume, while **Agent J** has completed the fewest tasks.  
- To better contextualize task counts, we can further explore the **number of tasks per day per agent**.  
    """)
     
        ## ----------------------------------------------------- ##
        
        ## ---------------- Generate Task Category (#) for all the Agents
        st.markdown("##### Task(#) By Task categories  (% Distribution)")
        st.markdown("In this section we will explore Task(#) distribution by Task categories")
        gen_percentage_plt_plotly(df, 'Task_Category')
        st.markdown("""
##### 💡 High-Level Task category Distribution Insights:
    
- **Research** is the most frequently performed task category, accounting for approximately **39.4%** of all activities.  
- The top three task types are related to **Research**, **Follow-up Calls**, and **Consent/Documents**, indicating the **core focus areas for agents**.
    """)
        ## -----------------------------------------------------
        
        
        ## ---------------- Generate Request Type (#) for all the Agents
        st.markdown("##### Task(#) By Request Type (% Distribution)")
        st.markdown("In this section we will explore % Task(#) distribution by Request Types")
        gen_percentage_plt_plotly(df, 'REQUEST_TYPE')
        st.markdown("""
    ##### 💡 Request Type Distribution Insights:
    
    - **Verification of Employment and Income (VOEI)** is the most common request type handled by agents.  
    - **Verification of Employment (VOE)** ranks second in frequency.  
    - **Self-Employed (SE)** tasks are the least common, indicating minimal volume in this category.  
    - **Re-Verification of Employment (RVOE)** accounts for only **2.3%**, suggesting that re-verification requests are relatively rare.
    """)
        ## -----------------------------------------------------
        
        
        ## ---------------- Generate Outcome Distribution (#) for all the Agents
        st.markdown("#####  Outcome Distribution")
        st.markdown("In this section we will explore % Task(#) distribution by Outcome category")
        gen_percentage_plt_plotly(df, 'Outcome_Category' )
        st.markdown("""
    ##### 💡 Outcome Distribution Insights:
    
    - Approximately **46.1%** of tasks result in **successful completion**, making it the most common outcome.  
    - The occurrence of **Rejections**, **No Matches**, and **Employer/Employee-related issues** is relatively low, indicating **fewer escalations or failures** in task resolution.
    """)    
        ## -----------------------------------------------------
    
        ## ---------------- Generate Handle Time  Distribution (#) for all the Agents
        st.markdown("#####  Handle Time Distribution")
        plot_handle_time_boxplot(df)
        # Calculate the outlier precentages 
        handle_time_outlier_percent, handle_time_indexes =  calculate_outlier_stats(df, 'HANDLE_TIME')
        
         
        st.markdown("""
    ##### 💡 Outcome Distribution Insights:
    - The HANDLE_TIME distribution is left-skewed, suggesting that most tasks are completed relatively quickly, with a smaller number taking significantly longer.
    - A notable 16.29% of tasks (approximately 42,582 out of 261,369) are identified as outliers, highlighting potential exceptions or process inefficiencies worth further investigation.""")
    
    st.markdown("---") 
    st.subheader(" 📈 2. Exploratory Analysis: Correlation Between Features")
    st.markdown("This section explores relationships between features using Cramér’s V for categorical variables.")
    with st.expander("🔽 Click to view trend analysis & plots"):
    
        st.markdown("#####  Correlation Analysis") 
        st.markdown("""
The goal of this section is to explore relationships between features and identify any strong associations.  
Understanding feature correlations can help uncover hidden patterns, reduce redundancy, and support more effective analysis and modeling.
""")
        # add the categorical var
        categorical_vars = ['AGENT_ID', 'REQUEST_TYPE', 'Task_Category', 'Outcome_Category']
        plot_categorical_associations_dython(df, categorical_vars)
         
        st.markdown("""
##### 💡 Correlation Analysis amongst Categorical Variables:

- Task Category and Agent ID – indicating that certain agents consistently handle specific types of tasks.
- Task Category and Outcome Category – suggesting that the nature of the task influences the likelihood of a particular outcome.
""")

        
        st.markdown("#####  Agent ID and Task Category Analysis")
        plot_agent_task_category_distribution(df)
        st.markdown("""
##### 💡 There is a strong relationship between Agent ID and Task Category:
- Certain agents appear to specialize in or consistently perform specific types of tasks.
- For example, Agent A predominantly handles tasks categorized as Research.
- Agent B also focuses heavily on Research, followed by tasks related to Consent/Documents.""")
        
        
        st.markdown("#####  Outcome category and Task Category Analysis")
        plot_agent_task_vs_outcome_distribution(df)
        st.markdown("""
##### 💡 There is a clear link between Task Category and Outcome:
- Tasks involving Consent/Documents and Research, are more likely to result in successful completion.""")
       
    st.markdown("---") 
    ## -----------------------------------------------------
    st.subheader(" ⏱️  3. Time Series Analysis: Task Volume & Performance Trends")    
    st.markdown("Analyze task activity over time to uncover workload fluctuations, performance patterns, and operational stability.")
    with st.expander("🔽 Click to view trend analysis & plots"):

        st.markdown("#####  Handle Time Trend")
        df_grouped = df.groupby('DATE').agg({'TASK_COUNT': 'sum'}).reset_index()
        df_grouped.rename(columns={'DATE': 'Date_v'}, inplace=True)
        line_moving_average_plot_plotly(df_grouped, 'TASK_COUNT')
        
        st.markdown("""
##### 💡 High Volatility in Daily Task Volume:
    
- The raw TASK_COUNT line shows frequent and significant fluctuations, indicating irregular daily workload.
- Several sharp drops suggest potential non-working days or system anomalies.
- Stabilized Trend via Exponential Smoothing:
    - The orange dashed line (exponential smoothing) reveals an underlying stable trend, despite short-term variability.
    - This smoothed trendline helps clarify the general workload pattern, reducing noise from daily spikes or dips.
- Mid-March Peak Activity:
    - Noticeable increase in smoothed task volume during mid to late March, indicating a temporary surge in workload or demand.
    - This could be linked to external events (e.g., seasonal cycles, end-of-quarter activity).
- Stable Load Around 3000 Tasks/Day:
    - The smoothed line consistently hovers around 2800–3200 tasks/day, suggesting that this is the baseline operational volume.
- Outlier Days with Sharp Dips:
    - Days with task counts dropping below 1000 (or close to 0) may be weekends, holidays, or system-related issues.
    - These should be flagged for review or excluded from performance analysis if not representative.""")
        
    
    st.markdown("---") 
    # ------------------- KPI Section ------------------- #
    st.subheader("📊 4  KPI Analysis - Agent-Level Insights")
    st.markdown("""
This section focuses on key performance indicators (KPIs) computed at the agent level to evaluate productivity and effectiveness.  
Metrics such as **average handle time**, **success rate**, **task volume**, and **tasks per active day** are analyzed to highlight performance trends and identify standout or struggling agents.

Visualizations below help compare agents across multiple dimensions to support data-driven decision-making in workforce management.
""")
   
    avg_tasks_per_day = np.round(df['TASK_COUNT'].sum() / df['DATE'].nunique(), 2)
    avg_handle_time = np.round(df['HANDLE_TIME'].sum() / df['TASK_COUNT'].sum(), 2)
    success_ratio = np.round((df['Outcome_Category'] == 'Successful Completion').sum() / len(df), 2)

    st.write(
        f"""
        i.   On average, **{avg_tasks_per_day}** tasks are handled per day.  
        ii.  Agents take approximately **{avg_handle_time} seconds** to complete a task.  
        iii. The overall task success ratio is **{success_ratio}**.
        """
    )
    with st.expander("🔽 Click to view trend analysis & plots"):
        st.markdown("""##### 📋 Key Metrics Summary  
-  Average Tasks per Day : Number of tasks completed daily across all agents  
-  Average Handle Time (sec) : Overall average time spent per task  
-  Avg Handle Time per Agent : Time efficiency measured individually per agent
        """) 
        
        agent_kpis = compute_agent_kpis(df)
        st.markdown("##### 📋 Agent KPI Summary (Sortable)")
        st.dataframe(agent_kpis[['Total_Tasks','Avg_Handle_Time','Median_Handle_Time','Task_Variance','Success_Rate','Avg_Tasks_Per_Day']].sort_values(by='Avg_Handle_Time', ascending=False).reset_index())
        
        st.markdown("##### 📊 Agent Performance: Avg Handle Time vs Success Rate")
        st.markdown("""
This dual-axis chart visualizes each agent's performance across two critical dimensions:
- **Average Handle Time** (bar): Represents the average time taken to complete tasks.
- **Success Rate** (line): Reflects the proportion of tasks completed successfully.

By combining these metrics, the chart highlights agents who maintain high efficiency without compromising task quality, and flags potential areas for performance coaching or process improvement.
""")
        sort_option = st.selectbox(
            "Sort agents by:",
            ['Avg_Handle_Time', 'Success_Rate', 'Total_Tasks', 'Median_Handle_Time', 'Avg_Tasks_Per_Day'],
            index=0
        )
        
    
        plot_handle_time_vs_success_rate_plotly(agent_kpis, sort_option)
        
        st.markdown("##### 📊 Agent Productivity vs. Efficiency")
    
        st.markdown("""
        This scatter plot visualizes the relationship between each agent's average task volume per day and the average time spent handling each task. 
        It helps identify high-performing agents who balance both speed and task throughput.
        """)
        plot_agent_productivity(agent_kpis)

    st.markdown("---") 
    # ------------------- Statistical Analysis ------------------- #
    st.subheader("🧪  5.  Statistical Testing")
    st.markdown("""ANOVA (Analysis of Variance): To determine if there are statistically significant differences in Avg_Handle_Time, Success_Rate, or Avg_Tasks_Per_Day across agents.""")
    with st.expander("🔽 Click to view the analysis"):
        anova_result = run_anova(df)
        
        st.write(f" ANOVA - F: {anova_result.statistic:.2f}, p-value: {anova_result.pvalue:.4f} ")
    
        if anova_result.pvalue < 0.05:
            st.success("Significant differences found between agent KPI's")
    
    st.markdown("---") 

    # ------------------- Clustering ------------------- #
    st.subheader("🔍 6. Agent Segmentation Summary (KMeans Clustering)")

    st.markdown("""
    This section summarizes agent segmentation results using **KMeans clustering** based on key performance metrics.  
    Agents are grouped into distinct clusters reflecting differences in productivity, efficiency, and task success rates.
    
    The goal is to identify patterns across agent behavior and provide tailored **recommendations** for coaching, recognition, or process improvement.
    """)
    with st.expander("🔽 Click to view trend analysis & plots"):
        clustered_df, score = run_kmeans_clustering(agent_kpis)
     
        st.markdown("#####  Visualize the Clusters")
        plot_cluster_scatter_plotly(clustered_df)
               
        clustered_df['Cluster_category'] = clustered_df['Cluster'].replace(clusters_mapping) 
        clustered_df['Performace Category'] = clustered_df['Cluster_category'].replace( cluster_dict)
        clustered_df['Recommendation'] = clustered_df['Cluster_category'].replace(cluster_rec)
#        st.subheader("📋 KPI Summary by Agent")
#        st.dataframe( 
#            clustered_df[[
#                'Performace Category', 'Total_Tasks', 'Avg_Handle_Time', 'Median_Handle_Time',\
#                    'Success_Rate', 'Avg_Tasks_Per_Day'
#        ]].sort_values(by='Performace Category').reset_index()
#    )
    
    # Section Header
    st.markdown("##### 💡 Agent   Characteristics & Recommendations")
    
    st.markdown("""
    This table summarizes the characteristics and suggested actions for each agent cluster based on performance indicators.  
    Use these insights to inform coaching strategies, performance interventions, or task reallocation.
    """)
    clustered_df['Agent'] = clustered_df.index
    clustered_df.reset_index(inplace = True)

    data_agent = pd.DataFrame(clustered_df[['Cluster_category','Recommendation','AGENT_ID' ]].\
                  groupby(['Cluster_category','Recommendation'])['AGENT_ID' ].apply(','.join).reset_index())
    # Display nicely in Streamlit
    st.dataframe(data_agent.style.set_properties(**{
        'text-align': 'left',
        'white-space': 'pre-wrap'
    }).set_table_styles([{
        'selector': 'th',
        'props': [('text-align', 'left')]
    }]), use_container_width=True)
        
    # Section Header for the Summary Table
    st.subheader("📊 7. Report Summary ")
    
    st.markdown("""
    This section provides a concise overview of key metrics derived from the agent performance dataset.  
    It captures task volume, agent coverage, success rates, and task-type distribution patterns.
    """)
    
    # Create summary data
    summary_data = {
        "Metric": [
            "Total Tasks Analyzed",
            "Total Agents",
            "Analysis Period",
            "Avg Tasks per Day",
            "Successful Task Completion Rate",
            "Outlier Tasks (Handle Time)",
            "Most Frequent Task Category",
            "Most Frequent Request Type",
            "Least Frequent Request Type",
            "Re-Verification Task Share"
        ],
        "Value": [
            Total_task,
            Total_Agents,
            str(Total_No_Of_Days) +" days",
            "~"  + str(avg_tasks_per_day),
            str(success_ratio * 100 ) + " %",
            str(len(handle_time_indexes)) + "( " + str(np.round(handle_time_outlier_percent,2))+ "% )",
            "Research (39.4%)",
            "VOEI",
            "SE",
            "2.30%"
        ]
    }  
    
    #, 

    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Display table in Streamlit
    
    with st.expander("📌 View Summary Insights"):
        st.markdown("#### 📋 Dataset Summary Overview")
        st.dataframe(summary_df.style.set_properties(**{
            'text-align': 'left',
            'font-size': '14px',
            'white-space': 'pre-wrap'
        }).set_table_styles([{
            'selector': 'th',
            'props': [('text-align', 'left')]
        }]), use_container_width=True)
    # [Insert your summary table code here]
        st.markdown("""
        - **Align Tasks with Agent Strengths**  
          Assign agents to task categories where they consistently demonstrate high efficiency and success rates.  
          *Example: Agent A excels in Research tasks — consider prioritizing similar assignments.*
    
        - **Monitor and Address Handle Time Outliers**  
          Approximately **16% of tasks** exhibited unusually long handle times.  
          These should be reviewed for bottlenecks, training gaps, or workflow inefficiencies.
    
        - **Leverage Clustering for Targeted Support**  
          KMeans segmentation identified **three distinct agent performance clusters**.  
          Use these insights to tailor coaching strategies, recognize high performers, and support underperformers effectively.
    
        - **Capitalize on High-Success Task Types**  
          Tasks related to **Employee Input, Consent/Documents, and Research** are strongly associated with successful completions.  
          Prioritize these categories in future optimization efforts.
    
        - **Maintain Balanced Workload Distribution**  
          The team handled an average of ~**3,000 tasks per day**.  
          Ensure balanced task assignment to prevent overload and sustain performance across all agents.
        """)
except Exception as main_error:
    st.error("🚨 An unexpected error occurred while running the dashboard.")
    st.exception(main_error)
