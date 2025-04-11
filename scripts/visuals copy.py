import seaborn as sns 
from matplotlib import pyplot  as plt 
from statsmodels.tsa.api import SimpleExpSmoothing
import pandas as pd
import streamlit as st


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

# function  
def line_moving_average_plot(pd_data, column_name, title = "default" ):
    '''
    This function plots the Data in columns along with 
    their moving averages(simple exp smoothing or rolling averages )
  
    Output:
        Plots for the column
    
    '''
    if title == "default" :
        title = column_name
        
 
    # Plt definition 
    fig, ax = plt.subplots(1,1, figsize = (19,6))
    ax.plot(pd_data['Date_v'], pd_data[column_name], label=column_name, linestyle='-') 
    # Smoothened Data 
    smooth_date = returnSmoothData(type_smooth="simpleexpsmoothing", data=pd_data[column_name])
    # plot the moving average
    ax.plot(pd_data['Date_v'],smooth_date ,linewidth=2, linestyle='-.',
                     alpha = 0.5, label=title + "- simple exp smoothing")

    # set the y label and Tittle
    ax.set_ylabel(column_name)
    ax.set_title("{} distribution over time".format(title.replace("_", " ")) )
    ax.grid()
    ax.legend() 
    fig.tight_layout()
    st.pyplot(fig) 

def gen_count_plt(df_DataFrame, strcol, image_w=9, image_h=6, font1 = 10, font2 = 8, horizontal=False):
 
    fig, ax = plt.subplots(figsize=(image_w, image_h))

    # Sort categories by count
    order = df_DataFrame[strcol].value_counts().index
    
    if horizontal:
        sns.countplot(y=strcol, data=df_DataFrame, order=order, ax=ax, palette='viridis')
        ax.set_ylabel(strcol, fontsize=font2)
        ax.set_xlabel('Number of Tasks', fontsize=font2)
    else:
        sns.countplot(x=strcol, data=df_DataFrame, order=order, ax=ax, palette='viridis')
        ax.set_xlabel(strcol, fontsize=font2)
        ax.set_ylabel('Number of Tasks', fontsize=font2)
        plt.xticks(rotation=45, ha='right')

    # Titles and labels
    ax.set_title(f'{strcol} Distribution', fontsize=font1, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

def gen_percentage_plt(df, strcol, image_w=10, image_h=6, horizontal=False, font1=12, font2=10):
    fig, ax = plt.subplots(figsize=(image_w, image_h))

    # Calculate counts and percentages
    counts = df[strcol].value_counts()
    percentages = counts / counts.sum() * 100
    order = percentages.index

    if horizontal:
        sns.barplot(x=percentages.values, y=order, palette='viridis', ax=ax)

        for p in ax.patches:
            width = p.get_width()
            ax.annotate(f'{width:.1f}%', 
                        (width, p.get_y() + p.get_height() / 2), 
                        ha='left', va='center', fontsize=font1)

        ax.set_xlabel('Percentage (%)', fontsize=font1)
        ax.set_ylabel(strcol, fontsize=font1)
    else:
        sns.barplot(x=order, y=percentages.values, palette='viridis', ax=ax)

        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.1f}%', 
                        (p.get_x() + p.get_width() / 2, height), 
                        ha='center', va='bottom', fontsize=font1)

        ax.set_xlabel(strcol, fontsize=font1)
        ax.set_ylabel('Percentage (%)', fontsize=font1)
        ax.tick_params(axis='x', rotation=45)

    ax.set_title(f'{strcol} Distribution', fontsize=font2, fontweight='bold')
    fig.tight_layout()
    st.pyplot(fig)

def plot_agent_heatmap(tukey_result):
    """Generate a symmetric heatmap of significant differences from Tukey HSD."""
    # Step 1: Initialize the matrix
    agents = list(tukey_result.groupsunique)
    sig_matrix = pd.DataFrame(False, index=agents, columns=agents)

    # Step 2: Fill the matrix with significance results
    for i in range(len(tukey_result.reject)):
        g1 = tukey_result._results_table.data[i + 1][0]
        g2 = tukey_result._results_table.data[i + 1][1]
        if tukey_result.reject[i]:
            sig_matrix.loc[g1, g2] = True
            sig_matrix.loc[g2, g1] = True

    # Step 3: Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(sig_matrix, cmap="Reds", cbar=False, linewidths=0.5, annot=True, fmt="", ax=ax)
    ax.set_title("Tukey HSD - Significant Differences Between Agents", fontsize=14)
    ax.set_xlabel("Agent")
    ax.set_ylabel("Agent")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    fig.tight_layout()

    # Step 4: Display in Streamlit
    st.pyplot(fig)
    
def plot_handle_time_vs_success_rate(agent_features, sort_by='Avg_Handle_Time'):
    """Displays a dual-axis chart for Avg Handle Time and Success Rate."""
    agent_data = agent_features.sort_values(by=sort_by, ascending=False)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bar chart: Avg Handle Time
    bar = ax1.bar(agent_data.index, agent_data['Avg_Handle_Time'], color='grey', label='Avg Handle Time (sec)')
    ax1.set_ylabel('Avg Handle Time (sec)', fontsize=12)
    ax1.set_xlabel('Agent ID', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)

    # Line chart: Success Rate
    ax2 = ax1.twinx()
    line = ax2.plot(agent_data.index, agent_data['Success_Rate'], color='red', marker='o', label='Success Rate ratio')
    ax2.set_ylabel('Success Rate Ratio', fontsize=12)

    # Title & legend
    fig.suptitle(f'Agent Performance: Avg Handle Time vs Success Rate (sorted by {sort_by})',
                 fontsize=14, fontweight='bold')
    fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    fig.tight_layout()
    st.pyplot(fig)

def plot_cluster_scatter(agent_features):
    """Visualizes agent clusters on PCA-reduced dimensions."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatterplot of PCA results
    sns.scatterplot(
        x='PCA1',
        y='PCA2',
        data=agent_features,
        hue='Cluster',
        palette='viridis',
        s=100,
        ax=ax
    )

    # Annotate each point with the Agent ID (index)
    for i in range(agent_features.shape[0]):
        ax.text(
            agent_features['PCA1'].iloc[i] + 0.05,
            agent_features['PCA2'].iloc[i],
            str(agent_features.index[i]),
            fontsize=8
        )

    ax.set_title('Agent Clusters Based on Performance')
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    fig.tight_layout()

    st.pyplot(fig)
