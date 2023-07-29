# Amazon Stock Price Prediction

we have built a Facebook Prophet Machine learning model in order to forecast the price of Amazon 30 days into the future. We have utilised plotly express for data visualization and evaluated the forecast using google finance in google sheets. Further, we have automated the entire stock forecasting process, enabling instantaneous forecast for any preferred stock.


Created Facebook Prophet Model

```bash
from prophet import Prophet
m = Prophet()
m.fit(prophet_df)

```
Forecasting

```bash
 future=m.make_future_dataframe(periods=30)
forecast=m.predict(future)
forecast     
```
Using Plotly for data visualization
```bash
px.line(forecast, x='ds', y='yhat')
figure=m.plot(forecast,xlabel='ds',ylabel='y')
figure2=m.plot_components(forecast)
```
Downloading Forecasted Data
```bash
from google.colab import files
forecast.to_csv('forecast.csv')
files.download('forecast.csv')
```

![Screenshot 2023-07-29 210201](https://github.com/pratiksha2712/Amazon-Stock-Price-Prediction/assets/82393814/dcf4919d-bc53-4ffd-8798-71878693a9e0)



