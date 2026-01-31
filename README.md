# Sentiment Analysis Social Media

## 📌 Project Description
This project is a **Social Media Sentiment Analysis** application built with Python and Streamlit. It uses Machine Learning models to classify social media posts into three sentiment categories:
- 🟢 **Positive**
- 🔴 **Negative**
- ⚪ **Neutral**

The application provides a user-friendly interface to input text and receive real-time sentiment predictions along with confidence scores and probability distributions.

## 🚀 Features
- **Interactive Web Interface**: Built using Streamlit for easy interaction.
- **Multi-Model Support**: Choose between multiple trained models:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - Naive Bayes
  - Gradient Boosting
- **Real-time Analysis**: Instant sentiment classification of user input.
- **Visualizations**: interactive bar charts showing class probabilities using Altair.
- **Comprehensive Analysis**: Includes notebooks for Exploratory Data Analysis (EDA) and model training.

## 🛠️ Tech Stack
- **Language**: Python
- **Framework**: Streamlit
- **Libraries**:
  - Scikit-learn (Machine Learning)
  - Pandas & NumPy (Data Manipulation)
  - Altair (Visualization)
  - Pickle (Model Serialization)

## 📂 Project Structure
```
├── app.py                          # Main Streamlit application
├── EDA_Preprocessing.ipynb         # Notebook for Data Analysis & Preprocessing
├── Model_1_Logistic_Regression.ipynb
├── Model_2_Random_Forest.ipynb
├── Model_3_SVM.ipynb
├── Model_4_Naive_Bayes.ipynb
├── Model_5_Gradient_Boosting.ipynb # Model Training Notebooks
├── Prediction_System.ipynb         # Prediction logic testing
├── preprocess_data.py              # Data preprocessing script
├── synthetic_social_media_data.csv # Dataset
├── *.pkl                           # Saved Models & Vectorizers
└── README.md                       # Project Documentation
```

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/srijanghosh05/Sentiment_analysis_social_media.git
cd Sentiment_analysis_social_media
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
Ensure you have the required libraries installed. You can install them using pip:
```bash
pip install streamlit pandas numpy scikit-learn altair
```

## 🖥️ How to Start the App
To run the web application locally, execute the following command in your terminal:

```bash
streamlit run app.py
```

Once the command is running, your default web browser should open automatically to `http://localhost:8501`.

## 📊 Model Performance
The project includes multiple trained models. Based on our evaluation:
- **Support Vector Machine (SVM)** achieved the highest accuracy of **98.05%**.


## 📝 License
This project is open-source and available for educational purposes.
