# 🪑Furniture Sales Predictor

A Machine Learning-powered web application that predicts the sales volume of furniture products based on their title, price, and discounts. This project utilizes **Random Forest Regression**, **TF-IDF Text Analysis**, and **Streamlit** for deployment.

## 🚀 Live Demo
https://unni-furniture-pred.streamlit.app/
---

## 📊 Project Overview
In the competitive e-commerce landscape, pricing and presentation are key. This project analyzes a dataset of furniture listings to identify patterns that lead to higher sales. 

### Key Features:
* **Predictive Modeling:** Uses a Random Forest Regressor to forecast sales.
* **NLP Integration:** Extracts features from product titles using TF-IDF.
* **Interactive UI:** A simple web interface for real-time predictions.
* **Data Insights:** Analyzes the impact of discounts and "set" bundles on consumer behavior.

---

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Web Framework:** Streamlit
* **Tools:** SQL (Data Querying), Excel (Pivot Analysis)

---

## 📁 Repository Structure
```text
├── app.py                     # Streamlit web application script
├── furniture_model.pkl         # Trained Random Forest model
├── tfidf_vectorizer.pkl        # Fitted TF-IDF vectorizer
├── requirements.txt           # Required Python libraries
├── ecommerce_furniture.csv    # Project dataset
└── README.md                  # Project documentation
