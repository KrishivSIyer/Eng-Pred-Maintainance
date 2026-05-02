# AeroPredict - Engine RUL Predictive Maintenance Dashboard ✈️🔧

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-yellow)
![Flask](https://img.shields.io/badge/Flask-2.3-lightgrey)

AeroPredict is a comprehensive Machine Learning application designed to predict the **Remaining Useful Life (RUL)** of turbofan engines using time-series sensor data. 

It provides an interactive, beautiful web dashboard to compare the performance of **Random Forest** and **LSTM (Long Short-Term Memory)** models across various complexity levels of the NASA CMAPSS dataset.

## 🚀 Features

- **Model Comparison:** Head-to-head performance analysis (MAE and RMSE) between traditional Machine Learning (Random Forest) and Deep Learning (LSTM) approaches.
- **Live Inference Simulation:** Select an engine dataset (FD001 to FD004) and run live RUL predictions on a random test engine.
- **Interactive Dashboard:** A sleek, dark-themed UI built with HTML, CSS (Vanilla), and JavaScript, utilizing `Chart.js` for dynamic data visualization.
- **Robust Backend:** A Flask-based API serving trained models, dynamic data parsing, and sequence padding on the fly.

## 📊 The Datasets

This project uses the **NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset. It consists of multiple multivariate time-series datasets representing different engine degradation trajectories:

- **FD001:** Single fault mode (HPC Degradation), Single operating condition (Sea Level) - *Simple*
- **FD002:** Single fault mode, Six operating conditions - *Complex*
- **FD003:** Two fault modes, Single operating condition - *Complex*
- **FD004:** Two fault modes, Six operating conditions - *Highly Complex*

## 🛠️ Technology Stack

- **Frontend:** HTML5, CSS3, Vanilla JavaScript, Chart.js
- **Backend:** Python, Flask
- **Machine Learning:** Scikit-Learn (Random Forest Regressor)
- **Deep Learning:** TensorFlow / Keras (LSTM Networks)
- **Data Processing:** Pandas, NumPy, Joblib

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/aeropredict.git
   cd aeropredict
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask Application:**
   ```bash
   python app.py
   ```

5. **Access the Dashboard:**
   Open your browser and navigate to `http://127.0.0.1:5000`

## 📁 Project Structure

```text
├── app.py                  # Main Flask application and API routes
├── requirements.txt        # Python dependencies
├── data/
│   └── raw/                # NASA CMAPSS raw text files (train/test/RUL)
├── models/                 # Saved Random Forest (.pkl), LSTM (.keras), and Scalers
├── results/                # CSV files containing historical evaluation metrics
├── static/
│   ├── app.js              # Frontend logic and API integration
│   └── style.css           # UI styling and animations
├── templates/
│   └── index.html          # Main dashboard interface
├── randomforest.py         # Baseline RF model training script
├── rftrainer.py            # Advanced RF training across multiple datasets
├── lstm.py                 # Baseline LSTM model training script
└── lstmfinetune.py         # Advanced LSTM fine-tuning script for FD002-FD004
```

## 📈 Results Summary

As dataset complexity increases (introducing multiple operating conditions and fault modes in FD002 and FD004), the Deep Learning (LSTM) model's ability to retain temporal memory begins to shine, whereas traditional Random Forest struggles with the highly non-linear degradation paths.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📝 License
This project is open-source and available under the [MIT License](LICENSE).
