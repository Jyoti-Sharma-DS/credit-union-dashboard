from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import streamlit as st


def run_kmeans_clustering(agent_features, n_clusters=3, random_state=42):
    features = ['Total_Tasks','Avg_Tasks_Per_Day', 'Avg_Handle_Time', 'Task_Variance',
                'Unique_Tasks', 'Success_Rate', 'Tasks_With_Issues']

    X = agent_features[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    agent_features['Cluster'] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    agent_features['PCA1'] = principal_components[:, 0]
    agent_features['PCA2'] = principal_components[:, 1]
    
    score = silhouette_score(X_scaled, agent_features['Cluster'])

    return agent_features , score




def asses_clusters_quality(score):
    strResults = ""
    if score > 0.5:
        strResults = "→ Strong clustering structure."
    elif score > 0.3:
        strResults = "→ Moderate clustering quality."
    else:
        strResults = "→ Weak clustering — review feature space or cluster count."
    
    return strResults