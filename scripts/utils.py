import os
import random
import warnings
import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing
import pandas as pd
from dython.nominal import associations,  cramers_v
import streamlit as st
 
def load_data(filepath):
    try:
        if not os.path.exists(filepath):
            st.warning(f"⚠️ File not found: `{filepath}`")
            return None
        df = pd.read_csv(filepath)
        if df.empty:
            st.warning("⚠️ The loaded file is empty. Please check your dataset.")
            return None
        return df
    except pd.errors.ParserError:
        st.error("❌ Error parsing CSV. Please ensure it's valid.")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error during data load: {str(e)}")
        return None
    

def to_camel_case(text):
    words = text.split()
    camel_cased_words = [words[0].capitalize()] + [word.capitalize() for word in words[1:]]
    return " ".join(camel_cased_words)


# function 
def fun_set_config_param (random_seed=1212):
    ''' fun_set_config_param:
    Input: 
    This function takes one input random_seed
    random_seed: Is a number which will be used for 
                configuring random number for the
                project configuration
    '''
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
    
def set_global_seed(random_seed=1212):
    """Set random seed for full reproducibility across libraries."""
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    warnings.filterwarnings('ignore')  

# function   
def returnSmoothData(type_smooth, data):
    ''' returnSmoothData:
    Input: 
        This function takes below inputs
        type_smooth: What type of smoothing needs to 
                    be performed. default is rolling
                    average window size 7.
        data: Data which needs to be Smoothened
    Output:
    return the Smoothened Data
    '''
    # Convert into lower type 
    type_smooth = type_smooth.lower()
    
    #Check 
    if (type_smooth == "simpleexpsmoothing"):
        model = SimpleExpSmoothing(data).fit(smoothing_level =0.1, optimized=True)
        # generate the smoothened values
        smooth_val = model.fittedvalues
    else:
        # Calculate the moving averages
        smooth_val = data.rolling(7).mean()
    
    return smooth_val


# Function returning outlier percentage and column name
def calculate_outlier_stats( dataframename, col_name ):
    
    ''' Calculating the outlier statistics:
    Input: 
    This function takes two arguments dataframename & col_name 
    where,
    i. dataframename: name of dataframe containing the numerical
                        data column.
    ii. col_name: name of the colum containing numerical data.
    Output:
    This function returns two values percent_value & index
    where,
    i. percent_value: % of records attributing to outliers.
    ii. index: boolean list indicating if the element at a 
                particular index is outlier or not.
    '''
    
    # Store the summary statitics
    summary_stat = dataframename[col_name].describe()
    
    # Calculate IQR range
    iqr_range = summary_stat['75%'] - summary_stat['25%']

    # Calculate the minimum limit
    min_limit  = summary_stat['25%'] - 1.5 * iqr_range
    
    # Calculate the maximum limit
    max_limit  = summary_stat['75%'] + 1.5 * iqr_range

    # Set the index values
    index = (dataframename[col_name] < min_limit) | (dataframename[col_name] > max_limit)
    
    # Calculate percentage  value
    percent_value = (sum(index)/len(index)) * 100
    
    # outlier percentage
    print("outlier percentage {:.3f} %".format(percent_value))
    
    # Return values
    return percent_value, index      

 

def compute_association_matrix(df, cat_vars):
    
    matrix = pd.DataFrame(np.zeros((len(cat_vars), len(cat_vars))),
                          index=cat_vars, columns=cat_vars)
    for i, var1 in enumerate(cat_vars):
        for j, var2 in enumerate(cat_vars):
            if i == j:
                matrix.loc[var1, var2] = 1.0
            else:
                matrix.loc[var1, var2] =  np.round( cramers_v(df[var1], df[var2]), 4)
    return matrix




