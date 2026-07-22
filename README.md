# Geospatial Valuation via Spatial Embeddings

## Project Overview

This project focuses on predicting real estate prices using geospatial information and machine learning. The objective is to improve property valuation by incorporating spatial relationships between houses.

---

## Tech Stack

- Python
- Pandas
- NumPy
- GeoPandas
- Folium
- Scikit-learn
- XGBoost
- Matplotlib

---

# Week 1 – Geospatial Data Acquisition and Processing

## Completed Tasks

- Downloaded King County Housing Dataset
- Loaded dataset using Pandas
- Performed Exploratory Data Analysis (EDA)
- Checked missing values
- Checked duplicate records
- Removed extreme price outliers using IQR
- Applied feature normalization using StandardScaler
- Created GeoDataFrame using GeoPandas
- Built an interactive map using Folium
- Saved cleaned dataset

### Outputs

- Cleaned Dataset
- Normalized Dataset
- Interactive Folium Map

---

# Week 2 – Feature Engineering and Baseline ML

## Completed Tasks

- Loaded cleaned dataset
- Created House Age feature
- Calculated Distance to City Center using Haversine Distance
- Selected features for training
- Performed Train-Test Split
- Trained XGBoost Baseline Regressor
- Generated predictions
- Evaluated model using RMSE and MAPE
- Saved trained model
- Saved prediction results
- Saved evaluation metrics

### Baseline Model Performance

- RMSE: **74709.82**
- MAPE: **11.93%**

### Baseline Model Limitations

- Uses only tabular features.
- Does not capture spatial relationships.
- Ignores neighbouring property influence.
- Performance may reduce in rapidly changing neighbourhoods.

---

## Project Structure
