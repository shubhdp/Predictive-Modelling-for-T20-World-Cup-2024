# Predictive-Modelling-for-T20-World-Cup-2024

## Project Overview

This project predicts the **winner of the T20 World Cup 2024** using machine learning models. It leverages historical cricket match data and team performance statistics to make predictions. The goal is to provide **data-driven insights** on team performance and match outcomes.

---

## Features

* Predicts match winners using **Random Forest** and **Deep Learning** models.
* Analyzes historical team performance metrics like runs, wickets, and player statistics.
* Generates visual insights using **charts and graphs**.
* Extendable to include player injuries, weather, toss outcomes, etc.

---

## Dataset

* **File:** `t20_worldcup_team_features_large_1000.csv`
* **Description:** Historical T20 World Cup match data with team and player statistics.
* **Columns include:** Runs, wickets, average scores, strike rates, and more.

> *Dataset is included in the repository for convenience.*

---

## Technology Stack

* **Language:** Python 3.x
* **Libraries:**

  * `pandas` & `numpy` – Data manipulation
  * `scikit-learn` – Random Forest machine learning
  * `tensorflow` / `keras` – Deep Learning
  * `matplotlib` & `seaborn` – Visualization

---

## Project Structure

```
Predictive-Modelling-for-T20-World-Cup-2024/
│
├── data/
│   └── t20_worldcup_team_features_large_1000.csv
│
├── models/
│   ├── random_forest_model.pkl
│   └── deep_learning_model.h5
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── scripts/
│   ├── train_random_forest.py
│   ├── train_deep_learning.py
│   └── predict_winner.py
│
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/shubhdp/Predictive-Modelling-for-T20-World-Cup-2024.git
cd Predictive-Modelling-for-T20-World-Cup-2024
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the models (optional)

* **Random Forest**

```bash
python scripts/train_random_forest.py
```

* **Deep Learning**

```bash
python scripts/train_deep_learning.py
```

### 4. Make predictions

```bash
python scripts/predict_winner.py
```

The output will display the **predicted winner** based on the trained model.

---

## Results

* **Random Forest Accuracy:** ~88%
* **Deep Learning Accuracy:** ~90%
* Model evaluation and visualizations are available in the `notebooks` folder.

---

## Future Improvements

* Include **player-specific statistics** like form and injuries.
* Integrate **real-time match conditions** such as weather, pitch, and toss.
* Develop a **web interface** for live match winner prediction.
* Experiment with other models like **XGBoost, LightGBM, or Ensemble methods**.

---

## Acknowledgements

* Historical cricket datasets.
* Open-source Python libraries (`pandas`, `scikit-learn`, `tensorflow`, `matplotlib`).
* Inspiration from sports analytics and predictive modeling projects.

---

✅ **Check out more of my projects:** [shubhdp GitHub](https://github.com/shubhdp)
