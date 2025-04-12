# 📈 Agent Performance Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://credit-union-agent-performance-analytics-js.streamlit.app/)

An interactive dashboard designed to evaluate agent productivity, task efficiency, and performance patterns within a credit union's employment verification call center.

---

## 📌 Project Overview

A credit union call center manages **employment verification requests**, each involving multiple task types.  
Agents may interact with the same order multiple times based on complexity and need. These interactions vary by type, handle time, and outcome.

This project analyzes **task-level operational data over 85 days** to uncover performance trends, inefficiencies, and optimization opportunities.

### 🎯 Objectives:
- Understand workload distribution across agents
- Identify performance gaps and process inefficiencies
- Evaluate success rates and handle times
- Detect agent specialization patterns

The final deliverable is a fully interactive dashboard that visualizes these insights, enabling smarter workforce planning and performance management.

---

## 📊 Executive Summary: Agent Performance Analysis

This dashboard evaluates **261,369 task records** completed by **15 agents** over **85 days**. The analysis reveals key patterns in productivity, task outcomes, and agent clustering.

### 🔍 Key Insights

#### ✅ Workload Distribution
- Significant variation in task volumes per agent
- "Research" is the most frequent task category

#### ✅ Performance Metrics
Agent-level KPIs include:
- Average tasks per day
- Average handle time
- Task success rate
- Frequency of issue-related outcomes

#### 📈 Trend Analysis
- Daily workload fluctuates but averages around **~3,000 tasks/day**
- Task completion patterns remain generally stable

#### 🔗 Correlation Insights
- Agents specialize in specific task types
- Certain task categories are more likely to result in successful outcomes

#### 👥 Clustering & Segmentation
Using **KMeans clustering**, agents were grouped into:
- **High performers**
- **Balanced performers**
- **Agents needing support**

These insights support **targeted coaching, resource planning,** and **process improvement**.

---

## 🖥️ Interactive Dashboard

Explore the full analysis live:

🔗 [Launch Dashboard →](https://credit-union-agent-performance-analytics-js.streamlit.app/)

---

## 🔮 Future Enhancements

- Integrate an LLM (e.g., ChatGPT) to auto-generate insights from plots  
- Incorporate forecasting models to predict workload trends  
- Add agent demographic or tenure data to deepen performance analysis

---

---

## 🧾 Final Summary

This project delivers a comprehensive view of agent performance by analyzing task-level data from a credit union’s employment verification operations.

### 📋 Dataset Summary Overview

| **Metric**                          | **Value**                   |
|------------------------------------|-----------------------------|
| Total Tasks Analyzed               | 261,369                     |
| Total Agents                       | 15                          |
| Analysis Period                    | 85 days                     |
| Avg Tasks per Day                  | ~3,074.93                   |
| Successful Task Completion Rate    | 46.0%                       |
| Outlier Tasks (Handle Time)        | 261,369 (16.29%)            |
| Most Frequent Task Category        | Research (39.4%)            |
| Most Frequent Request Type         | VOEI                        |
| Least Frequent Request Type        | SE                          |
| Re-Verification Task Share         | 2.30%                       |

---

### 💡 Key Takeaways & Recommendations

- **Align Tasks with Agent Strengths**  
  Assign agents to task categories where they consistently demonstrate high efficiency and success rates.  
  *Example: Agent A excels in Research tasks — prioritize such assignments.*

- **Monitor and Address Handle Time Outliers**  
  Approximately **16% of tasks** had unusually long handle times.  
  These should be reviewed to identify potential bottlenecks, training gaps, or process inefficiencies.

- **Leverage Clustering for Targeted Support**  
  **KMeans clustering** identified three agent segments: high performers, balanced performers, and underperformers.  
  Use this segmentation to customize coaching, recognition, and workload balancing.

- **Capitalize on High-Success Task Types**  
  Task types such as **Employee Input, Consent/Documents, and Research** show strong success rates.  
  Focus optimization and training around these categories.

- **Maintain Balanced Workload Distribution**  
  The team processes around **~3,000 tasks/day**.  
  Avoid overloading high-performing agents to sustain productivity across the board.

---
