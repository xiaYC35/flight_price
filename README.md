

# ✈️ A Dynamic Pricing and Forecasting System for Airline Fares

## 📌 Project Description
Airline ticket prices are highly dynamic and often unpredictable, creating uncertainty for both airlines and travelers.  

This project focuses on analyzing historical flight data from **Delhi to major Indian cities** (e.g., Mumbai, Bangalore, Hyderabad) to better understand pricing mechanisms and build predictive models. An interactive flight price prediction system built with XGBoost and Streamlit. This project leverages historical flight data to predict ticket prices in real time based on key features such as airline, destination, booking time, flight duration, and cabin class.

It introduces a dual-model architecture, where separate models are trained for Economy and Business class to better capture their distinct pricing behaviors.

The main research questions include:
- **What factors drive ticket prices?**  
  (e.g., airline brand, departure time, days before departure)
- **Is there an optimal booking time?**  
  When can travelers purchase tickets at the lowest price?
- **Can we predict future airfares?**  
  Develop models to forecast price trends and ranges.

---
# 🚀 Project Highlights

## 🧠 Dual-Model Architecture

Two independent XGBoost regression models are trained for:

Economy Class
Business Class

---

## 📊 Interactive Visualization

A clean and intuitive **Streamlit web app** enables real-time predictions without any coding.

Users can easily adjust flight parameters and instantly see predicted prices.

---

## ⚙️ Automated Feature Engineering

The system includes a robust preprocessing pipeline:

* One-Hot Encoding for categorical features
* Automatic feature alignment between training and inference
* Handling unseen categories safely

This resolves common production issues such as:

> feature names mismatch errors

---

## 🚨 Anomaly Detection

Residual-based analysis is applied during training to identify abnormal pricing patterns and improve model robustness.

---


## 🚀 Key Features
- 📊 Exploratory Data Analysis (EDA) on airline pricing patterns  
- 🔍 Feature importance analysis (airline, timing, advance booking, etc.)  
- 🤖 Machine Learning models for airfare prediction  
- 📈 Price trend visualization and insights  

---

# 🛠️ Tech Stack

* **Core ML:** XGBoost, Scikit-Learn
* **Data Processing:** Pandas, NumPy
* **Web Framework:** Streamlit
* **Visualization:** Matplotlib

---

## 📦 Installation

```bash
git clone https://github.com/xiaYC35/flight_price.git
cd flight_price
pip install -r requirements.txt

---

# ⚡ Quick Start

## 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

---

## 2️⃣ Install dependencies

Make sure you have Python 3.7+ installed.

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available:

```bash
pip install streamlit pandas numpy xgboost scikit-learn joblib
```

---

## 3️⃣ Prepare model files

Ensure the following trained model files are in the project root:

```
model_economy.pkl
model_business.pkl
feature_cols.pkl
```

These are generated during training.

---

## 4️⃣ Run the application

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

# 🧭 How to Use

## 1. Select Cabin Class

Choose between:

* Economy
* Business

---

## 2. Enter Flight Details

* Airline (e.g., Vistara, Air India)
* Destination
* Days before journey
* Flight duration
* Number of stops

---

## 3. Make Prediction

Click **"Predict Price"**

---

## 4. View Result

The system will display:

* Predicted fare
* Booking advice (e.g., *Early Bird*, *Last-minute*)

---

# 🧠 Model Logic

## 🔀 Routing Strategy

The system routes input to different models based on cabin class:

* Economy → Economy model
* Business → Business model

---

## 🔧 Feature Alignment

To ensure robustness in production:

* Input features are One-Hot encoded
* Missing features are automatically filled with `0`
* Ensures compatibility with training feature space

This prevents:

> ValueError: feature_names mismatch

---

## 🤖 Independent Prediction

Each model learns its own pricing distribution:

* Economy → price-sensitive patterns
* Business → premium pricing behavior

---

# 📂 Dataset

This project uses a **flight price dataset (e.g., Kaggle Flight Price Dataset)** containing:

* Airline
* Source & Destination
* Departure time
* Duration
* Stops
* Booking time
* Fare

---

# 📄 License

This project is licensed under the MIT License.
