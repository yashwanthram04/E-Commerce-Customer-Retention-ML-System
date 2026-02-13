
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Churn Prediction System",
    page_icon="🔮",
    layout="wide"
)

# Load model and artifacts
@st.cache_resource
def load_model_artifacts():
    """Load model and preprocessing artifacts"""
    try:
        model = joblib.load('churn_prediction_model_final.pkl')
        scaler = joblib.load('feature_scaler.pkl')

        with open('feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]

        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, scaler, feature_names = load_model_artifacts()

# Prediction function
def predict_churn(customer_data):
    """Make churn prediction"""
    try:
        df = pd.DataFrame([customer_data])

        # Ensure all features
        for fname in feature_names:
            if fname not in df.columns:
                df[fname] = 0
        df = df[feature_names]

        # Predict
        prediction = int(model.predict(df)[0])
        probability = float(model.predict_proba(df)[0, 1])

        # Risk level
        if probability >= 0.7:
            risk_level = "High"
            risk_color = "red"
        elif probability >= 0.4:
            risk_level = "Medium"
            risk_color = "orange"
        else:
            risk_level = "Low"
            risk_color = "green"

        return {
            'prediction': prediction,
            'probability': probability,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'confidence': max(probability, 1 - probability)
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# App Header
st.title("🔮 Customer Churn Prediction System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 Navigation")
    page = st.radio("Select Page", ["Single Prediction", "Batch Prediction", "Model Info"])

    st.markdown("---")
    st.header("ℹ️ About")
    st.write("This system predicts customer churn using machine learning.")
    st.write("**Model Version:** v2.1")

# PAGE 1: Single Prediction
if page == "Single Prediction":
    st.header("👤 Single Customer Prediction")

    # Create input form
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Information")
        customer_id = st.text_input("Customer ID", value="C001")
        segment = st.selectbox("Customer Segment",
                              ['Regular', 'Premium', 'VIP', 'New'])

        st.subheader("Order History")
        num_orders = st.number_input("Number of Orders", min_value=0, value=5)
        total_spent = st.number_input("Total Spent ($)", min_value=0.0, value=1250.0)
        avg_order_value = st.number_input("Avg Order Value ($)", min_value=0.0, value=250.0)
        max_order_value = st.number_input("Max Order Value ($)", min_value=0.0, value=450.0)

    with col2:
        st.subheader("Engagement Metrics")
        days_since_last = st.number_input("Days Since Last Order", min_value=0, value=45)
        customer_lifetime_days = st.number_input("Customer Lifetime (days)",
                                                 min_value=0, value=365)

        st.subheader("RFM Scores (1-5)")
        recency_score = st.slider("Recency Score", 1, 5, 3)
        frequency_score = st.slider("Frequency Score", 1, 5, 3)
        monetary_score = st.slider("Monetary Score", 1, 5, 3)

    # Predict button
    if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
        # Prepare customer data
        customer_data = {
            'customer_id': customer_id,
            'num_orders': num_orders,
            'total_spent': total_spent,
            'avg_order_value': avg_order_value,
            'max_order_value': max_order_value,
            'days_since_last_order': days_since_last,
            'customer_lifetime_days': customer_lifetime_days,
            'recency_score': recency_score,
            'frequency_score': frequency_score,
            'monetary_score': monetary_score
        }

        # Make prediction
        result = predict_churn(customer_data)

        if result:
            st.markdown("---")
            st.header("📊 Prediction Results")

            # Display results in columns
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Prediction",
                         "CHURN" if result['prediction'] == 1 else "ACTIVE",
                         delta=None)

            with col2:
                st.metric("Churn Probability",
                         f"{result['probability']:.1%}")

            with col3:
                if result['risk_level'] == 'High':
                    st.error(f"🚨 {result['risk_level']} Risk")
                elif result['risk_level'] == 'Medium':
                    st.warning(f"⚠️ {result['risk_level']} Risk")
                else:
                    st.success(f"✅ {result['risk_level']} Risk")

            with col4:
                st.metric("Confidence", f"{result['confidence']:.1%}")

            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = result['probability'] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Churn Probability"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': result['risk_color']},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgreen"},
                        {'range': [40, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))

            st.plotly_chart(fig, use_container_width=True)

            # Recommendations
            st.subheader("💡 Recommended Actions")

            if result['risk_level'] == 'High':
                st.error("""
                **URGENT: Immediate retention action required!**
                - Contact customer within 24 hours
                - Offer personalized retention incentive (discount, loyalty points)
                - Assign to retention specialist
                - Schedule follow-up call
                """)
            elif result['risk_level'] == 'Medium':
                st.warning("""
                **Monitor and engage proactively:**
                - Send personalized email campaign
                - Offer special promotion
                - Monitor activity closely
                - Schedule check-in next week
                """)
            else:
                st.success("""
                **Maintain regular engagement:**
                - Continue standard communication
                - Encourage product reviews
                - Promote new products
                - Nurture loyalty program participation
                """)

# PAGE 2: Batch Prediction
elif page == "Batch Prediction":
    st.header("📁 Batch Customer Prediction")

    st.write("Upload a CSV file with customer data to predict churn for multiple customers.")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded_file is not None:
        # Read CSV
        try:
            df = pd.read_csv(uploaded_file)

            st.subheader("📊 Uploaded Data Preview")
            st.dataframe(df.head(10))

            if st.button("🔮 Predict for All Customers", type="primary"):
                # Make predictions
                predictions = []

                progress_bar = st.progress(0)

                for idx, row in df.iterrows():
                    result = predict_churn(row.to_dict())
                    if result:
                        predictions.append({
                            'customer_id': row.get('customer_id', f'C{idx}'),
                            'churn_prediction': 'Churn' if result['prediction'] == 1 else 'Active',
                            'churn_probability': result['probability'],
                            'risk_level': result['risk_level']
                        })

                    progress_bar.progress((idx + 1) / len(df))

                # Display results
                results_df = pd.DataFrame(predictions)

                st.success(f"✅ Predictions complete for {len(results_df)} customers!")

                # Summary metrics
                col1, col2, col3 = st.columns(3)

                with col1:
                    churn_count = (results_df['churn_prediction'] == 'Churn').sum()
                    st.metric("Predicted Churners", churn_count)

                with col2:
                    churn_rate = (results_df['churn_prediction'] == 'Churn').mean()
                    st.metric("Churn Rate", f"{churn_rate:.1%}")

                with col3:
                    high_risk = (results_df['risk_level'] == 'High').sum()
                    st.metric("High Risk Customers", high_risk)

                # Results table
                st.subheader("📊 Prediction Results")
                st.dataframe(results_df)

                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

                # Visualization
                fig = px.pie(results_df, names='risk_level',
                           title='Risk Level Distribution',
                           color='risk_level',
                           color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'})
                st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error processing file: {e}")

# PAGE 3: Model Info
elif page == "Model Info":
    st.header("ℹ️ Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Model Details")
        st.write(f"**Model Type:** {type(model).__name__ if model else 'Unknown'}")
        st.write(f"**Version:** v2.1")
        st.write(f"**Features:** {len(feature_names) if feature_names else 0}")

        st.subheader("📊 Performance Metrics")
        st.write("**Test Set Performance:**")
        st.write("- Accuracy: 87.3%")
        st.write("- Precision: 84.1%")
        st.write("- Recall: 79.2%")
        st.write("- F1-Score: 81.6%")
        st.write("- ROC-AUC: 89.4%")

    with col2:
        st.subheader("🎯 Risk Level Definitions")

        st.error("**High Risk (≥70%)**")
        st.write("Immediate retention action required")

        st.warning("**Medium Risk (40-70%)**")
        st.write("Monitor and engage proactively")

        st.success("**Low Risk (<40%)**")
        st.write("Maintain regular engagement")

    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        st.subheader("⭐ Top 10 Important Features")

        importance_df = pd.DataFrame({
            'Feature': feature_names[:10],
            'Importance': model.feature_importances_[:10]
        }).sort_values('Importance', ascending=False)

        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                    title='Feature Importance')
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("© 2025 Churn Prediction System | Powered by Machine Learning")
