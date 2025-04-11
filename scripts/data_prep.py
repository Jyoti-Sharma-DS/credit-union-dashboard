import pandas as pd

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df['TASK_DATE'] = pd.to_datetime(df['TASK_DATE'], errors='coerce')
    df['DATE'] = df['TASK_DATE'].dt.normalize()

    # Convert types
    df['AGENT_ID'] = df['AGENT_ID'].astype('string')
    df['TASK_TYPE'] = df['TASK_TYPE'].astype('string')
    df['OUTCOME'] = df['OUTCOME'].astype('string').str.strip()
    df['REQUEST_TYPE'] = df['REQUEST_TYPE'].astype('string')

    return df


def map_task_categories(df, task_type_mapping):
    # Define task type mapping dictionary
    df['Task_Category'] = df['TASK_TYPE'].map(task_type_mapping).astype('string')
    return df


def map_outcome_categories(df, outcome_mapping):
    df['Outcome_Category'] = df['OUTCOME'].replace(outcome_mapping).astype('string')
    return df
