# Predictive Analytics for Smart Bike-Sharing Optimization 🚴‍♀️📊

![Python](https://img.shields.io/badge/Python-3.9+-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Accuracy](https://img.shields.io/badge/Top%20Model%20Accuracy-93%25-brightgreen)

This project focuses on predicting bicycle demand in public bike-sharing systems using machine learning. The goal is to optimize bike availability across different hours and days, helping operators improve service efficiency and user satisfaction.

## Project Overview

Bicycle-sharing systems are essential for modern urban mobility, but efficiently managing inventory requires accurate demand prediction. This project leverages historical bike usage data, temporal features (hour, day, month), and weather indicators to forecast high and low demand periods.

## Features & Methodology

- **Data Preprocessing:** Handling missing values, feature engineering (weather index, binary time features), and standard scaling.
- **Machine Learning Models:** Implemented multiple classifiers including Logistic Regression, Random Forest, LDA, QDA, and K-Nearest Neighbors. Models were optimized using grid search and evaluated with F1-score metrics to handle class imbalance.
- **Evaluation:** Weighted and macro F1-scores were used along with overall accuracy to ensure robust performance across high and low bike demand classes.

## Key Results

- **Random Forest:** Accuracy 93%, F1-weighted 92%, F1-macro 86% – best balanced performance across classes.  
- **KNN:** Accuracy 92%, F1-weighted 92%, F1-macro 86%.  
- **Other Models:** Logistic Regression, LDA, and QDA achieved up to 90% accuracy.  
- **Naive Baseline:** Accuracy 50%, highlighting the significant improvement from ML models.

## Model Performance 📊

### Accuracy Comparison
![Accuracy Comparison](images/model_accuracy.png)

### F1-Weighted Score Comparison
![F1-Weighted](images/f1_weighted.png)

### F1-Macro Score Comparison
![F1-Macro](images/f1_macro.png)


## Repository Structure
 <pre>
HumayraMusarrat/
├── data_analysis.py    # Exploratory Data Analysis & feature engineering
├── training.py         # Model training and hyperparameter tuning
├── test.py             # Model evaluation on test data
├── README.md           # Project overview and instructions
</pre>

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/HumayraMusarrat/YourRepoName.git
```

2. Install dependancies

   pip install -r requirements.txt

3. Run the script

   python data_analysis.py
   python training.py
   python test.py

## Future Work
Extend to multiple cities for better generalization
Explore advanced deep learning models for superior accuracy

## Contact
Humayra Musarrat – humayramusarrat89@gmail.com

