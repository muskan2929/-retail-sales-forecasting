# Retail Sales Forecasting

## Business Problem
Retailers lose revenue due to overstocking and understocking.
This project predicts daily store sales to optimize inventory.

## Dataset
Rossmann Store Sales (Kaggle) — 1M+ rows, 1115 stores

## Models Built
| Model             | RMSE   | R²   |
|------------------|--------|------|
| Linear Regression | ~2800  | 0.82 |
| Ridge Regression  | ~2650  | 0.84 |
| Lasso Regression  | ~2700  | 0.83 |

## Tech Stack
Python, scikit-learn, pandas, matplotlib, seaborn

## Key Insights
- Promotions are the strongest sales driver
- Weekend and month-end patterns are significant
