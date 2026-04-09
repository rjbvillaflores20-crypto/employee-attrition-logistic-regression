# Employee Attrition Prediction using Logistic Regression (MATLAB)

## 📌 Overview
This project demonstrates the formulation, implementation, and application of a **Binary Logistic Regression model** to predict employee attrition.

The model estimates the probability that an employee leaves an organization using workplace-related features.

---

## 🎯 Objectives
- Formulate a logistic regression model
- Implement Maximum Likelihood Estimation (MLE)
- Apply Newton–Raphson / IRLS algorithm
- Evaluate classification performance

---

## 🧮 Mathematical Model

The probability model:

P(Y=1|X) = 1 / (1 + exp(-(β₀ + β₁X₁ + ... + βₖXₖ)))

Logit form:

log(p / (1 - p)) = β₀ + β₁X₁ + ... + βₖXₖ

---

## ⚙️ Implementation

Two approaches are used:
1. MATLAB built-in (`glmfit`)
2. Manual Newton–Raphson method

---

## Dataset

IBM HR Analytics Employee Attrition Dataset

- Source: IBM Sample Data (public dataset)
- File: `WA_Fn-UseC_-HR-Employee-Attrition.csv`

Features used:
- Age  
- Monthly Income  
- Years at Company  
- Job Satisfaction  
- (Optional) Environment Satisfaction, Work-Life Balance  

Target Variable:
- Attrition (Yes = 1, No = 0)

## ▶️ How to Run

1. Open MATLAB
2. Navigate to `/employ`
3. Run:

```matlab
main
