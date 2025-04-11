import numpy as np

def compute_agent_kpis(df):
    active_days = df.groupby('AGENT_ID')['DATE'].nunique()
    issue_categories = ['Contact Issues', 'Employer/Employee Issues']

    agent_features = df.groupby('AGENT_ID').agg(
        Total_Tasks=('TASK_COUNT', 'sum'),
        Avg_Handle_Time=('HANDLE_TIME', 'mean'),
        Max_Handle_Time=('HANDLE_TIME', 'max'),
        Min_Handle_Time=('HANDLE_TIME', 'min'),
        Median_Handle_Time=('HANDLE_TIME', 'median'),
        Task_Variance=('HANDLE_TIME', 'std'),
        Unique_Tasks=('Task_Category', 'nunique'),
        Success_Rate=('Outcome_Category', lambda x: (x=='Successful Completion').mean()),
        Tasks_With_Issues=('Outcome_Category', lambda x: x.isin(issue_categories).mean())
    ).fillna(0)

    agent_features['Avg_Tasks_Per_Day'] = agent_features['Total_Tasks'] / active_days
    agent_features['Avg_Tasks_Per_Day'] = agent_features['Avg_Tasks_Per_Day'].fillna(0)

    return agent_features.round(2)