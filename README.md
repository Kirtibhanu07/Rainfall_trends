# Rainfall Trends Visualization, Anomaly Detection & Forecasting

## 📌 Overview
This project aims to visualize and detect anomalies in historical rainfall data and forecast future rainfall trends using machine learning techniques such as **Isolation Forest**, **K-Means Clustering**, and **ARIMA Time Series Forecasting**.

## 🔍 Features
- **Anomaly Detection**: Identifies anomalous years with extreme rainfall and drought using Isolation Forest.
- **Clustering**: Groups years into 'Dry', 'Normal', and 'Wet' categories using K-Means clustering.
- **Time Series Forecasting**: Predicts rainfall trends for the next 30 years using ARIMA.
- **Visualization**: Generates interactive scatter plots to display patterns in rainfall data.

## 🛠 Technologies Used
- **Python**
- **Pandas**
- **NumPy**
- **Plotly**
- **Scikit-learn**
- **Statsmodels**
- **Matplotlib**

## 📂 Dataset
The dataset contains annual and seasonal rainfall data for different years, with columns such as:
- `YEAR` – Year of record
- `ANNUAL` – Total annual rainfall (mm)
- `Jan-Feb`, `Mar-May`, `Jun-Sep`, `Oct-Dec` – Seasonal rainfall values

## 🚀 Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/rainfall_trends.git
   cd rainfall_trends
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```sh
   python rainfall_analysis.py
   ```

## 📊 Usage
### **1. Detect Rainfall Anomalies**
```python
from sklearn.ensemble import IsolationForest

isolation_forest = IsolationForest(contamination=0.05, random_state=42)
rainfall_data['Annual_Anomaly'] = isolation_forest.fit_predict(rainfall_data[['ANNUAL']])
```

### **2. K-Means Clustering for Rainfall Patterns**
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
kmeans = KMeans(n_clusters=3, random_state=42)
rainfall_data['Rainfall_Cluster'] = kmeans.fit_predict(scaler.fit_transform(rainfall_data[['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec', 'ANNUAL']]))
```

### **3. Forecast Future Rainfall Trends (Next 30 Years)**
```python
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

model = ARIMA(rainfall_data.set_index("YEAR")["ANNUAL"], order=(3,1,2))
model_fit = model.fit()
forecast = model_fit.forecast(steps=30)
```

## 📈 Results & Visualization
The project includes **interactive plots** for:
- **Anomalous Years** (Extreme Rainfall & Drought)
- **Clustered Rainfall Patterns**
- **Future Rainfall Forecast (Prophet, ARIMA)**

---

💡 **Contributions Welcome!** If you find this useful, feel free to open an issue or submit a PR.

🌟 **Star the repo if you like it!** ⭐

