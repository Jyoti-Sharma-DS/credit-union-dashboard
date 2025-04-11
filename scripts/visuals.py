import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot  as plt 
import plotly.figure_factory as ff
from statsmodels.tsa.api import SimpleExpSmoothing
from scripts.utils import returnSmoothData, compute_association_matrix,to_camel_case
 

# function  
def line_moving_average_plot(pd_data, column_name, title = "default" ):
    '''
    This function plots the Data in columns along with 
    their moving averages(simple exp smoothing or rolling averages )
  
    Output:
        Plots for the column
    
    '''
    column_name_label = to_camel_case(column_name.replace("_", " "))
    if title == "default" :
        title = column_name_label
        
    
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
    column_name_label = to_camel_case(strcol.replace("_", " "))
    if horizontal:
        sns.countplot(y=strcol, data=df_DataFrame, order=order, ax=ax, palette='viridis')
        ax.set_ylabel(column_name_label, fontsize=font2)
        ax.set_xlabel('Number of Tasks', fontsize=font2)
    else:
        sns.countplot(x=strcol, data=df_DataFrame, order=order, ax=ax, palette='viridis')
        ax.set_xlabel(column_name_label, fontsize=font2)
        ax.set_ylabel('Number of Tasks', fontsize=font2)
        plt.xticks(rotation=45, ha='right')

    # Titles and labels
    ax.set_title(f'{column_name_label} Distribution', fontsize=font1, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)


def gen_percentage_plt(df, strcol, image_w=10, image_h=6, horizontal=False, font1=12, font2=10):
    
    column_name_label = to_camel_case(strcol.replace("_", " "))
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
        ax.set_ylabel(column_name_label, fontsize=font1)
    else:
        sns.barplot(x=order, y=percentages.values, palette='viridis', ax=ax)

        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.1f}%', 
                        (p.get_x() + p.get_width() / 2, height), 
                        ha='center', va='bottom', fontsize=font1)

        ax.set_xlabel(column_name_label, fontsize=font1)
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
    
    

###########################################################################################
##  ----------------------------------- Plotly  -------------------------------------------


def line_moving_average_plot_plotly(pd_data, column_name, title="default"):
    
    column_name_label = to_camel_case(column_name.replace("_", " "))
    if title == "default":
        title = column_name_label

    smooth_data = returnSmoothData("simpleexpsmoothing", pd_data[column_name])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pd_data['Date_v'], y=pd_data[column_name],
                             mode='lines', name=column_name))
    fig.add_trace(go.Scatter(x=pd_data['Date_v'], y=smooth_data,
                             mode='lines', name=f"{title} (Smoothed)",
                             line=dict(dash='dash', width=2, color='orange')))

    fig.update_layout(
        title=f"{title.replace('_', ' ')} Distribution Over Time",
        xaxis_title="Date",
        yaxis_title=column_name,
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)
 
    
    
def gen_count_plt_plotly(df, strcol, horizontal=False, color = 'viridis'):
    
    column_name_label = to_camel_case(strcol.replace("_", " "))
    counts = df[strcol].value_counts().reset_index()
    counts.columns = [strcol, 'Count']

    fig = px.bar(
        counts,
        x='Count' if horizontal else strcol,
        y=strcol if horizontal else 'Count',
        orientation='h' if horizontal else 'v',
        color='Count',
        color_continuous_scale = color
    )

    fig.update_layout(
        title=f"{column_name_label} Distribution",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)
    
    
    
    
def gen_percentage_plt_plotly(df, strcol, horizontal=False, color = 'viridis'):
    
    column_name_label = to_camel_case(strcol.replace("_", " "))
    
    percentages = df[strcol].value_counts(normalize=True) * 100
    plot_df = percentages.reset_index()
    plot_df.columns = [strcol, 'Percentage']
 

    fig = px.bar(
        plot_df,
        x='Percentage' if horizontal else strcol,
        y=strcol if horizontal else 'Percentage',
        orientation='h' if horizontal else 'v',
        text='Percentage',
        color='Percentage',
        color_continuous_scale=color
    )

    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        title=f"{column_name_label} Percentage Distribution",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)



def plot_agent_heatmap_plotly(tukey_result):
    agents = list(tukey_result.groupsunique)
    sig_matrix = pd.DataFrame(False, index=agents, columns=agents)

    for i in range(len(tukey_result.reject)):
        g1 = tukey_result._results_table.data[i + 1][0]
        g2 = tukey_result._results_table.data[i + 1][1]
        if tukey_result.reject[i]:
            sig_matrix.loc[g1, g2] = True
            sig_matrix.loc[g2, g1] = True

    fig = px.imshow(sig_matrix.astype(int), text_auto=True, color_continuous_scale='Reds')
    fig.update_layout(
        title="Tukey HSD: Significant Differences Between Agents",
        xaxis_title="Agent",
        yaxis_title="Agent",
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    
    
def plot_handle_time_vs_success_rate_plotly(agent_features, sort_by='Avg_Handle_Time'):
    agent_data = agent_features.sort_values(by=sort_by, ascending=False).reset_index()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=agent_data['AGENT_ID'],
        y=agent_data['Avg_Handle_Time'],
        name='Avg Handle Time (sec)',
        yaxis='y',
        marker_color='gray'
    ))

    fig.add_trace(go.Scatter(
        x=agent_data['AGENT_ID'],
        y=agent_data['Success_Rate'],
        name='Success Rate',
        yaxis='y2',
        mode='lines+markers',
        line=dict(color='red')
    ))

    fig.update_layout(
        title=f"Agent Performance: Avg Handle Time vs Success Rate (sorted by {sort_by})",
        yaxis=dict(title='Avg Handle Time (sec)'),
        yaxis2=dict(title='Success Rate Ratio', overlaying='y', side='right'),
        xaxis_title='Agent ID',
        legend=dict(x=0.8, y=1.15, orientation="h"),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_cluster_scatter_plotly(agent_features):
    fig = px.scatter(
        agent_features.reset_index(),
        x='PCA1',
        y='PCA2',
        color='Cluster',
        hover_data=['AGENT_ID'],
        text='AGENT_ID',
        title='Agent Clusters Based on Performance',
        color_continuous_scale='viridis',
        height=600,
        width=800
    )

    fig.update_traces(textposition='top center', marker=dict(size=10))
    fig.update_layout(legend_title_text='Cluster', margin=dict(t=50, l=20, r=20, b=20))

    st.plotly_chart(fig, use_container_width=True)



def plot_handle_time_with_outliers(df, value_col='HANDLE_TIME', date_col='DATE', outlier_mask=None):
    df = df.copy()
    df['Type'] = 'Normal'

    if outlier_mask is not None:
        df.loc[outlier_mask, 'Type'] = 'Outlier'

    fig = px.scatter(
        df,
        x=date_col,
        y=value_col,
        color='Type',
        title='Handle-Time( Sec) Over Time with Outliers Highlighted',
        color_discrete_map={
            'Normal': 'steelblue',
            'Outlier': 'red'
        },
        opacity=0.8,
        height=500
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Handle Time (sec)",
        legend_title="Data Type"
    )

    st.plotly_chart(fig, use_container_width=True)
 

def plot_categorical_associations_dython(df , categorical_vars ):
    """
    Uses dython's associations function to compute pairwise associations (Cramér's V) 
    for the given categorical variables, then visualizes the resulting matrix as an 
    interactive Plotly heatmap.
    
    Parameters:
        df (DataFrame): DataFrame containing your data.
        cat_vars (list): List of categorical variable names.
    """
    # Ensure the selected columns are treated as strings.
 
    
    assoc_matrix = compute_association_matrix(df, categorical_vars)
     
 
    # Check if we obtained a valid 2D matrix.
    if assoc_matrix is None or not hasattr(assoc_matrix, 'shape') or len(assoc_matrix.shape) != 2:
         st.error("Dython associations did not return a valid association matrix. "
                  "Please update dython or compute the association matrix manually.")
         return
    
    # Create an interactive heatmap with Plotly.
    fig = px.imshow(
         assoc_matrix,
         text_auto=True,
         color_continuous_scale='Agsunset',
         origin='lower',
         zmin=0,
         zmax=1,
         labels={'x': 'Variable', 'y': 'Variable'},

         height=500
    )
    fig.update_layout(
         title="Associations between Features ",
         xaxis_title="Variable",
         yaxis_title="Variable"
    )
    
    st.plotly_chart(fig, use_container_width=True)

 

def plot_agent_task_category_distribution(df):
    """
    Creates an interactive stacked bar chart showing the distribution of Task_Category for each AGENT_ID.
    
    Parameters:
        df (DataFrame): DataFrame containing at least 'AGENT_ID' and 'Task_Category' columns.
    """
    # Create a contingency table (cross-tabulation) of AGENT_ID and Task_Category counts
    ctab = pd.crosstab(df['AGENT_ID'], df['Task_Category'])
    
    # Convert counts to percentages (row-wise)
    ctab_pct = ctab.div(ctab.sum(axis=1), axis=0) * 100
    
    # Reset index so that AGENT_ID becomes a column and melt to long format
    ctab_pct = ctab_pct.reset_index()
    
    melted_df = pd.melt(ctab_pct, id_vars='AGENT_ID', var_name='Task_Category', value_name='Percentage')
    
    # Create a stacked bar chart using Plotly Express
    fig_agent_task_category = px.bar(
        melted_df,
        x='AGENT_ID',
        y='Percentage',
        color='Task_Category',
        title='Task Category Distribution by Agent',
        labels={'AGENT_ID': 'Agent ID', 'Percentage': '% of Tasks(#)'},
        opacity=0.8,
        height=900,
        width = 900
    )
    
    # Set bar mode to 'stack' and order the x-axis categories by total count
    fig_agent_task_category.update_layout(
        barmode='stack',
        xaxis={'categoryorder': 'total descending'}
    )
    
    # Display the interactive chart in Streamlit
    st.plotly_chart(fig_agent_task_category, use_container_width=True)


def plot_agent_task_vs_outcome_distribution(df):
    """
    Creates an interactive stacked bar chart showing the percentage distribution of Task_Category
    for each Outcome_Category.
    
    Parameters:
        df (DataFrame): DataFrame containing at least 'Outcome_Category' and 'Task_Category' columns.
    """
    # Create a contingency table (cross-tabulation) of Outcome_Category and Task_Category counts
    ctab = pd.crosstab(df['Outcome_Category'], df['Task_Category'])
    
    # Convert counts to percentages (row-wise)
    ctab_pct = ctab.div(ctab.sum(axis=1), axis=0) * 100
    
    # Reset index so that Outcome_Category becomes a column and melt to long format
    ctab_pct = ctab_pct.reset_index()
    melted_df = pd.melt(ctab_pct, id_vars='Outcome_Category', var_name='Task_Category', value_name='Percentage')
    
    # Create a stacked bar chart using Plotly Express
    fig_task_vs_outcome = px.bar(
        melted_df,
        x='Outcome_Category',
        y='Percentage',
        color='Task_Category',
        title='Task Category Distribution by Outcome',
        labels={'Outcome_Category': 'Outcome Category', 'Percentage': '% of Tasks(#)'},
        opacity=0.8,
        height=500
    )
    
    # Set bar mode to 'stack' and order the x-axis categories by total count (if needed)
    fig_task_vs_outcome.update_layout(
        barmode='stack',
        xaxis={'categoryorder': 'total descending'}
    )
    
    # Display the interactive chart in Streamlit
    st.plotly_chart(fig_task_vs_outcome, use_container_width=True)
 

def plot_handle_time_boxplot(df):
    # Create a boxplot using Plotly Express with a discrete color that pops on a dark background.
    fig = px.box(
        df,
        x='HANDLE_TIME',
        title="Handle Time Distribution",
        labels={"HANDLE_TIME": "Handle Time (sec)"},
        color_discrete_sequence=["#00CC96"]  # A bright green shade for contrast
    )
    
    # Update the layout to use a dark template and adjust fonts and background colors.
    fig.update_layout(
        template="plotly_dark",
        title_font=dict(size=18, family="Arial", color="white"),
        xaxis_title="Handle Time (sec)",
        xaxis=dict(color="white"),
        yaxis=dict(color="white"),
        paper_bgcolor="black",
        plot_bgcolor="black",
        margin=dict(t=50, l=50, r=50, b=50)
    )
    
    # Display the interactive plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    
def plot_agent_productivity(agent_data):
    """
    Creates an interactive scatter plot showing Avg Tasks Per Day vs Avg Handle Time for each agent,
    annotated with agent IDs.
    
    Parameters:
        agent_data (DataFrame): Must contain 'Avg_Tasks_Per_Day' and 'Avg_Handle_Time', indexed by agent ID.
    """
    # Reset index to include Agent IDs as a column
    df_plot = agent_data.reset_index().rename(columns={'index': 'Agent_ID'})
    
    fig = px.scatter(
        df_plot,
        x='Avg_Tasks_Per_Day',
        y='Avg_Handle_Time',
        text='AGENT_ID',  # or 'Agent_ID' based on actual column name
        title='Agent Productivity vs Efficiency',
        labels={
            'Avg_Tasks_Per_Day': 'Avg Tasks Per Day',
            'Avg_Handle_Time': 'Avg Handle Time (sec)'
        },
        width=800,
        height=500
    )
    
    # Improve hover and label layout
    fig.update_traces(textposition='top center', marker=dict(size=10), hovertemplate='%{text}<br>Tasks: %{x}<br>Time: %{y}')
    fig.update_layout(
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    # Render in Streamlit
    st.plotly_chart(fig, use_container_width=True)

