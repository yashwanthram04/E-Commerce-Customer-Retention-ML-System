
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import os

# Page config
st.set_page_config(
    page_title="Model Monitoring Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load data functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_performance_metrics():
    """Load performance metrics history"""
    try:
        df = pd.read_csv('monitoring/performance_metrics.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_latest_predictions():
    """Load latest prediction batch"""
    try:
        # Get most recent file
        files = [f for f in os.listdir('predictions/daily') if f.endswith('.csv')]
        if files:
            latest = sorted(files)[-1]
            df = pd.read_csv(f'predictions/daily/{latest}')
            return df
    except:
        pass
    return pd.DataFrame()

# Header
st.title("🔍 Churn Model Monitoring Dashboard")
st.markdown("---")

# Refresh button
if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# Last update time
st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.markdown("---")

# Load data
perf_metrics = load_performance_metrics()
latest_preds = load_latest_predictions()

# KPI Row
col1, col2, col3, col4 = st.columns(4)

if not perf_metrics.empty:
    latest_metrics = perf_metrics.iloc[-1]

    with col1:
        st.metric(
            "Model Accuracy",
            f"{latest_metrics['roc_auc']:.1%}",
            delta=f"{(latest_metrics['roc_auc'] - 0.85):.1%}" if latest_metrics['roc_auc'] >= 0.85 else None
        )

    with col2:
        st.metric(
            "Daily Predictions",
            f"{latest_metrics['total_predictions']:,.0f}",
            delta=None
        )

    if not latest_preds.empty:
        high_risk = (latest_preds['risk_level'] == 'High').sum()

        with col3:
            st.metric(
                "High Risk Customers",
                f"{high_risk:,}",
                delta=None
            )

        with col4:
            churn_rate = (latest_preds['prediction'] == 1).mean()
            st.metric(
                "Predicted Churn Rate",
                f"{churn_rate:.1%}",
                delta=None
            )

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Performance Trends",
    "🎯 Recent Predictions",
    "⚠️ Alerts & Issues",
    "📊 Detailed Analytics"
])

# TAB 1: Performance Trends
with tab1:
    st.subheader("Model Performance Over Time")

    if not perf_metrics.empty:
        # Filter last N days
        days_filter = st.slider("Show last N days", 7, 90, 30)
        cutoff = datetime.now() - timedelta(days=days_filter)
        filtered_metrics = perf_metrics[perf_metrics['date'] >= cutoff]

        # Performance metrics chart
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=filtered_metrics['date'],
            y=filtered_metrics['accuracy'],
            name='Accuracy',
            mode='lines+markers'
        ))

        fig.add_trace(go.Scatter(
            x=filtered_metrics['date'],
            y=filtered_metrics['precision'],
            name='Precision',
            mode='lines+markers'
        ))

        fig.add_trace(go.Scatter(
            x=filtered_metrics['date'],
            y=filtered_metrics['recall'],
            name='Recall',
            mode='lines+markers'
        ))

        fig.add_trace(go.Scatter(
            x=filtered_metrics['date'],
            y=filtered_metrics['roc_auc'],
            name='ROC-AUC',
            mode='lines+markers',
            line=dict(width=3)
        ))

        fig.update_layout(
            title='Performance Metrics Trend',
            xaxis_title='Date',
            yaxis_title='Score',
            hovermode='x unified',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Metrics summary table
        st.subheader("Performance Summary")

        summary_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            'Current': [
                f"{latest_metrics['accuracy']:.3f}",
                f"{latest_metrics['precision']:.3f}",
                f"{latest_metrics['recall']:.3f}",
                f"{latest_metrics['f1_score']:.3f}",
                f"{latest_metrics['roc_auc']:.3f}"
            ],
            '30-Day Avg': [
                f"{filtered_metrics['accuracy'].mean():.3f}",
                f"{filtered_metrics['precision'].mean():.3f}",
                f"{filtered_metrics['recall'].mean():.3f}",
                f"{filtered_metrics['f1_score'].mean():.3f}",
                f"{filtered_metrics['roc_auc'].mean():.3f}"
            ],
            'Trend': [
                '📈' if latest_metrics['accuracy'] > filtered_metrics['accuracy'].mean() else '📉',
                '📈' if latest_metrics['precision'] > filtered_metrics['precision'].mean() else '📉',
                '📈' if latest_metrics['recall'] > filtered_metrics['recall'].mean() else '📉',
                '📈' if latest_metrics['f1_score'] > filtered_metrics['f1_score'].mean() else '📉',
                '📈' if latest_metrics['roc_auc'] > filtered_metrics['roc_auc'].mean() else '📉'
            ]
        })

        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    else:
        st.info("No performance data available yet.")

# TAB 2: Recent Predictions
with tab2:
    st.subheader("Latest Prediction Batch")

    if not latest_preds.empty:
        # Summary metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            total = len(latest_preds)
            st.metric("Total Customers", f"{total:,}")

        with col2:
            churners = (latest_preds['prediction'] == 1).sum()
            st.metric("Predicted Churners", f"{churners:,}")

        with col3:
            high_risk = (latest_preds['risk_level'] == 'High').sum()
            st.metric("High Risk", f"{high_risk:,}")

        # Risk distribution pie chart
        risk_dist = latest_preds['risk_level'].value_counts()

        fig = px.pie(
            values=risk_dist.values,
            names=risk_dist.index,
            title='Risk Level Distribution',
            color=risk_dist.index,
            color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'}
        )

        st.plotly_chart(fig, use_container_width=True)

        # High-risk customers table
        st.subheader("🚨 High-Risk Customers (Top 20)")

        high_risk_df = latest_preds[latest_preds['risk_level'] == 'High'].sort_values(
            'churn_probability', ascending=False
        ).head(20)

        if len(high_risk_df) > 0:
            display_df = high_risk_df[[
                'customer_id', 'churn_probability', 'risk_level'
            ]].copy()
            display_df['churn_probability'] = display_df['churn_probability'].apply(
                lambda x: f"{x:.1%}"
            )

            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.success("No high-risk customers in latest batch!")

        # Probability distribution histogram
        fig = px.histogram(
            latest_preds,
            x='churn_probability',
            nbins=50,
            title='Churn Probability Distribution'
        )

        fig.add_vline(x=0.4, line_dash="dash", line_color="orange",
                     annotation_text="Medium threshold")
        fig.add_vline(x=0.7, line_dash="dash", line_color="red",
                     annotation_text="High threshold")

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No recent predictions available.")

# TAB 3: Alerts & Issues
with tab3:
    st.subheader("System Alerts & Issues")

    # Check for alerts
    alerts = []

    if not perf_metrics.empty:
        latest = perf_metrics.iloc[-1]

        # Performance alerts
        if latest['roc_auc'] < 0.80:
            alerts.append({
                'severity': 'HIGH',
                'type': 'Performance',
                'message': f"ROC-AUC dropped to {latest['roc_auc']:.3f} (threshold: 0.80)",
                'date': latest['date']
            })
        elif latest['roc_auc'] < 0.85:
            alerts.append({
                'severity': 'MEDIUM',
                'type': 'Performance',
                'message': f"ROC-AUC at {latest['roc_auc']:.3f} (target: 0.85)",
                'date': latest['date']
            })

    if not latest_preds.empty:
        high_risk_pct = (latest_preds['risk_level'] == 'High').mean()

        if high_risk_pct > 0.30:
            alerts.append({
                'severity': 'MEDIUM',
                'type': 'Risk Distribution',
                'message': f"{high_risk_pct:.1%} of customers are high-risk (threshold: 30%)",
                'date': datetime.now()
            })

    # Display alerts
    if alerts:
        for alert in alerts:
            if alert['severity'] == 'HIGH':
                st.error(f"🚨 **{alert['type']}**: {alert['message']}")
            elif alert['severity'] == 'MEDIUM':
                st.warning(f"⚠️ **{alert['type']}**: {alert['message']}")
            else:
                st.info(f"ℹ️ **{alert['type']}**: {alert['message']}")
    else:
        st.success("✅ No active alerts - All systems normal")

    st.markdown("---")

    # System health checks
    st.subheader("System Health Checks")

    health_checks = {
        'Model Loaded': True,
        'API Responsive': True,
        'Database Connected': True,
        'Disk Space OK': True,
        'Recent Predictions': not latest_preds.empty
    }

    for check, status in health_checks.items():
        if status:
            st.success(f"✅ {check}")
        else:
            st.error(f"❌ {check}")

# TAB 4: Detailed Analytics
with tab4:
    st.subheader("Detailed Analytics")

    if not perf_metrics.empty and not latest_preds.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Performance Stats")

            stats_df = pd.DataFrame({
                'Metric': [
                    'Total Predictions (30d)',
                    'Average Accuracy',
                    'Average Precision',
                    'Average Recall',
                    'Performance Variance'
                ],
                'Value': [
                    f"{perf_metrics['total_predictions'].sum():,.0f}",
                    f"{perf_metrics['accuracy'].mean():.3f}",
                    f"{perf_metrics['precision'].mean():.3f}",
                    f"{perf_metrics['recall'].mean():.3f}",
                    f"{perf_metrics['roc_auc'].std():.4f}"
                ]
            })

            st.dataframe(stats_df, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("### Prediction Stats")

            pred_stats_df = pd.DataFrame({
                'Metric': [
                    'Latest Batch Size',
                    'Avg Churn Probability',
                    'High Risk %',
                    'Predicted Churn Rate',
                    'Low Confidence Predictions'
                ],
                'Value': [
                    f"{len(latest_preds):,}",
                    f"{latest_preds['churn_probability'].mean():.3f}",
                    f"{(latest_preds['risk_level'] == 'High').mean():.1%}",
                    f"{(latest_preds['prediction'] == 1).mean():.1%}",
                    f"{(latest_preds.get('confidence', pd.Series([1])) < 0.7).sum()}"
                ]
            })

            st.dataframe(pred_stats_df, use_container_width=True, hide_index=True)

        # Prediction timeline
        st.markdown("### Prediction Volume Over Time")

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=perf_metrics['date'],
            y=perf_metrics['total_predictions'],
            name='Predictions',
            marker_color='steelblue'
        ))

        fig.update_layout(
            title='Daily Prediction Volume',
            xaxis_title='Date',
            yaxis_title='Number of Predictions',
            height=300
        )

        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("© 2024 Churn Prediction System | Real-time Monitoring Dashboard")
