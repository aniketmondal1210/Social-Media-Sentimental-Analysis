import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import string
import altair as alt

# -----------------------------------------------------------------------------
# MODEL LOADING SECTION
# -----------------------------------------------------------------------------

@st.cache_resource
def load_resources():
    """
    Load the vectorizer and label encoder.
    """
    try:
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        return vectorizer, le
    except FileNotFoundError:
        st.error("Essential files not found. Please ensure 'tfidf_vectorizer.pkl' and 'label_encoder.pkl' are present.")
        return None, None

@st.cache_resource
def load_model(model_name):
    """
    Load a specific model based on selection.
    """
    filename = f"model_{model_name}.pkl"
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Model file '{filename}' not found.")
        return None

# Load shared resources
vectorizer, le = load_resources()

# -----------------------------------------------------------------------------
# PREDICTION LOGIC
# -----------------------------------------------------------------------------

def clean_text(text):
    """Cleans the input text for the model."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text

def predict_sentiment(model, text):
    """
    Predicts the sentiment of the input text using the selected model.
    """
    if model is None or vectorizer is None or le is None:
        return {'sentiment': 'Error', 'confidence': 0.0, 'probabilities': {'Positive': 0, 'Negative': 0, 'Neutral': 0}}

    # 1. Clean the text
    cleaned_text = clean_text(text)
    
    # 2. Vectorize
    vec_text = vectorizer.transform([cleaned_text])
    
    # 3. Predict
    pred = model.predict(vec_text)[0]
    
    # Handle probability prediction (some models might not support it if not configured, but ours do)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec_text)[0]
    else:
        # Fallback for models without predict_proba (though all ours should have it)
        probs = np.zeros(len(le.classes_))
        probs[pred] = 1.0
        
    sentiment = le.inverse_transform([pred])[0]
    
    # Map probabilities to classes
    class_probs = {class_name: prob for class_name, prob in zip(le.classes_, probs)}
    
    # Ensure all classes are present
    for class_name in ['Positive', 'Negative', 'Neutral']:
        if class_name not in class_probs:
            class_probs[class_name] = 0.0
            
    confidence = np.max(probs)
    
    return {
        'sentiment': sentiment,
        'confidence': confidence,
        'probabilities': class_probs
    }

# -----------------------------------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------------------------------

# Page Config
st.set_page_config(
    page_title="Social Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Model Selection
    model_options = {
        "Logistic Regression": "lr",
        "Random Forest": "rf",
        "Support Vector Machine (SVM)": "svm",
        "Naive Bayes": "nb",
        "Gradient Boosting": "gb"
    }
    
    selected_model_name = st.selectbox(
        "Select Model",
        list(model_options.keys())
    )
    
    model_key = model_options[selected_model_name]
    
    # Load the selected model
    current_model = load_model(model_key)
    
    st.divider()
    
    st.title("ℹ️ Project Info")
    st.markdown(f"""
    **Social Sentiment Analyzer**
    
    Classifying posts into:
    *   🟢 **Positive**
    *   🔴 **Negative**
    *   ⚪ **Neutral**
    
    **Current Model:** {selected_model_name}\n
    **Support Vector Machine (SVM) has the highest accuracy of 98.05%**
    ---
    *Built with Streamlit & Python*
    """)

# Main Content
st.title("💬 Social Sentiment Analyzer")
st.markdown(f"Analyze sentiment using **{selected_model_name}**.")



# Text Input
user_text = st.text_area(
    "Enter your post here:",
    placeholder="e.g., I absolutely love this new feature! It's fantastic.",
    height=150
)

# Analyze Button
if st.button("Analyze Sentiment", type="primary"):
    if not user_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner(f"Analyzing with {selected_model_name}..."):
            # Call prediction function
            result = predict_sentiment(current_model, user_text)
            
            sentiment = result['sentiment']
            confidence = result['confidence']
            probs = result['probabilities']
            
            # Display Results
            st.markdown("### Result")
            
            # Determine color based on sentiment
            if sentiment == "Positive":
                color = "green"
                box_color = "#d4edda"
                text_color = "#155724"
            elif sentiment == "Negative":
                color = "red"
                box_color = "#f8d7da"
                text_color = "#721c24"
            else:
                color = "gray"
                box_color = "#e2e3e5"
                text_color = "#383d41"
            
            # Display Sentiment Box
            st.markdown(
                f"""
                <div style="background-color: {box_color}; color: {text_color}; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                    <h2 style="margin:0;">{sentiment}</h2>
                    <p style="margin:0;">Confidence: {confidence:.2%}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            

            
            # Columns for details
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Confidence Score")
                st.progress(confidence)
                st.caption(f"The model is {confidence:.2%} sure of this result.")
                
            with col2:
                st.markdown("#### Class Probabilities")
                # Create a DataFrame for the chart
                chart_data = pd.DataFrame(
                    list(probs.items()),
                    columns=['Sentiment', 'Probability']
                )
                
                # Display Bar Chart using Altair
                c = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X('Sentiment', sort=None),
                    y='Probability',
                    color=alt.Color('Sentiment', scale=alt.Scale(
                        domain=['Positive', 'Negative', 'Neutral'],
                        range=['#28a745', '#dc3545', '#6c757d']
                    )),
                    tooltip=['Sentiment', alt.Tooltip('Probability', format='.2%')]
                ).properties(
                    height=300
                )
                
                st.altair_chart(c, use_container_width=True)

# Footer
st.markdown("---")
st.caption("© 2025 Sentiment Analysis Project. All rights reserved.")
