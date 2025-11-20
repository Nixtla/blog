# Simple Anomaly Detection in Time Series via Optimal Baseline Subtraction (OBS)
**Anomaly detection** in time series is used to identify unexpected patterns in your time series, and it is widely applied in different fields. In **energy engineering**, a spike in power usage might signal a fault. In **finance**, sudden drops or peaks can indicate major market events. In **mechanical systems**, unusual vibrations may reveal early signs of failure. In this blogpost, we will use **weather data** as an example use case, and we will find the anomalies in temperature time series for different cities all over the world.  

## Optimal Baseline Subtraction (OBS) Description

### OBS Introduction

If you have a bank of time series and you want to understand if and in what portion of the time series you have an anomaly, a simple but very efficient metod is called **optimal baseline subtraction (OBS)**. OBS is based on comparing each time series segment to the most similar historical pattern and analyzing the difference to detect unexpected deviations. 

### OBS Algorithm

The OBS algorithm is the following:

- **Split the time series into individual segments**, where each segment represents a unit of repeated behavior (e.g., a day, a cycle, or a process run).
- **Build a library of historical segments** by collecting all previous segments in the time series bank.
- **Compare your target segment** with all other segments in the library using a similarity metric, such as Mean Absolute Error (MAE).
- **Select the most similar segment** from the library. We define the most similar segment as the **optimal baseline**.
- **Subtract the optimal baseline from the target segment** to isolate the residual (i.e., the absolute difference).
- **Analyze the residual** to identify large deviations, which are flagged as potential anomalies.

![Alt text](images/Workflow.png)



## Optimal Baseline Subtraction Application

### Script folder

You can find all the code and data you need in this [public github folder](https://github.com/PieroPaialungaAI/OptimalBaselineSubtraction.git), that you can download with:

```bash
git clone https://github.com/PieroPaialungaAI/OptimalBaselineSubtraction.git
```

### Data Source

The data used for this article originally come from an Open Database on Kaggle. You can find the original source of the dataset [here](https://www.kaggle.com/datasets/selfishgene/historical-hourly-weather-data). Nonetheless, note that **you don't need to download it again, as everything you need is in the ```OBS_Data``` folder.**

### Preprocessed Data

The "preprocessing" part of the data is handled by the ```data.py``` code, so we can just deal with the fun stuff here. If you want to see the specific code for every block, feel free to visit the source code. A table with the attributes for the cities can be found in the ```.city_attribute_data```:





```python
from data import *
data = TimeSeriesData()
data.city_attribute_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>City</th>
      <th>Country</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Vancouver</td>
      <td>Canada</td>
      <td>49.249660</td>
      <td>-123.119339</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Portland</td>
      <td>United States</td>
      <td>45.523449</td>
      <td>-122.676208</td>
    </tr>
    <tr>
      <th>2</th>
      <td>San Francisco</td>
      <td>United States</td>
      <td>37.774929</td>
      <td>-122.419418</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Seattle</td>
      <td>United States</td>
      <td>47.606209</td>
      <td>-122.332069</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Los Angeles</td>
      <td>United States</td>
      <td>34.052231</td>
      <td>-118.243683</td>
    </tr>
  </tbody>
</table>
</div>



While the correspoding time series can be found in the  ```.temperature_data```, where each column represents a city, and the ```.datetime``` is the time step column. For example, the data for the city of **Vancouver** are the following:


```python
data.temperature_data[['Vancouver','datetime']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Vancouver</th>
      <th>datetime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>284.630000</td>
      <td>2012-10-01 13:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>284.629041</td>
      <td>2012-10-01 14:00:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>284.626998</td>
      <td>2012-10-01 15:00:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>284.624955</td>
      <td>2012-10-01 16:00:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>284.622911</td>
      <td>2012-10-01 17:00:00</td>
    </tr>
  </tbody>
</table>
</div>



### Selecting the Target Segment

So let's say we have our dataset and we want to see if there is an anomaly in a specific section (target curve). The following code pulls the city of interest and allows you to pick a specific window (e.g. a day, a week or a month) for that specific city. For example, let's pick **day** number **377** for **city = Los Angeles**. This will be our target curve. In the **remaining** part of the dataset (everything but the target curve), we can look for the optimal baseline.


```python
city = 'Los Angeles'
segment_class = 'day'
segment_idx = 377
data.isolate_city_and_time(city = city, segment_idx = segment_idx, segment_class = segment_class)
data.plot_target_and_baseline()
```


    
![png](OBS_Notebook_files/OBS_Notebook_5_0.png)
    


### Selecting the Bank of Candidates

By selected our city, window and index, we have uniquely defined our **target segment/window/signal**. All the remaining windows form your bank of candidates. For example, for our **day** window, our segments have 24 points (one per hour). For this reason the list of candidates will have shape ```(number of days - 1, 24)```. 


```python
data.list_of_candidates.shape
```




    (1853, 24)



Let's display some random candidates.


```python
data.plot_target_and_candidates()
```


    
![png](OBS_Notebook_files/OBS_Notebook_9_0.png)
    


### Selecting the optimal baseline  
As we can see above, some baselines follow our target curve very well. Other baselines don't follow our target curve at all.
In this step, we will use the MAE metric to find the **optimal baseline** (i.e. time series that is the closest to our target in the list of candidates). 
The result of the Optimal Baseline search has been stored in a dictionary:



```python
optimal_baseline_data = data.find_optimal_baseline()
optimal_baseline_data
```




    {'optimal_baseline_curve': array([293.7 , 292.67, 291.21, 289.89, 288.9 , 288.12, 287.4 , 286.99,
            286.68, 286.2 , 285.71, 285.3 , 284.77, 284.55, 285.35, 287.65,
            290.15, 292.2 , 293.24, 295.04, 296.17, 296.54, 296.74, 296.53]),
     'optimal_baseline_diff': array([0.26      , 0.2485    , 0.51      , 0.1       , 0.02      ,
            0.11      , 0.42      , 0.14      , 0.24      , 0.07      ,
            0.25      , 0.35      , 0.38      , 0.93033333, 3.26066667,
            1.256     , 4.1655    , 0.79      , 1.46      , 0.4175    ,
            0.41      , 0.3       , 0.17      , 0.42      ]),
     'optimal_baseline_error': 0.694937499999997,
     'target_curve': array([293.96      , 292.4215    , 290.7       , 289.79      ,
            288.88      , 288.23      , 287.82      , 287.13      ,
            286.44      , 286.27      , 285.96      , 285.65      ,
            285.15      , 283.61966667, 282.08933333, 288.906     ,
            285.9845    , 291.41      , 294.7       , 294.6225    ,
            295.76      , 296.24      , 296.57      , 296.11      ])}



Our plot function can be used to display the optimal baseline vs the target curve and the **residual**, which is the absolute difference between the twos divided by the maximum of the target curve (scaling factor):


```python
data.plot_target_and_optimal_baseline()
```


    
![png](OBS_Notebook_files/OBS_Notebook_13_0.png)
    


### Anomaly Detection Algorithm
We can see that the target time series is very in line with the Optimal Baseline, except for a small area (around the 15th hour), that we can consider as an **anomaly**. So now we can use a threshold to flag anomalies: any point where the residual exceeds this threshold is considered anomalous, as shown in the graph below.


```python
data.run_anomaly_detection(threshold=0.007, plot = True)
```




    {'time': array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23]),
     'residual': array([8.76690157e-04, 8.37913477e-04, 1.71966146e-03, 3.37188522e-04,
            6.74377044e-05, 3.70907374e-04, 1.41619179e-03, 4.72063931e-04,
            8.09252453e-04, 2.36031965e-04, 8.42971305e-04, 1.18015983e-03,
            1.28131638e-03, 3.13697722e-03, 1.09945937e-02, 4.23508784e-03,
            1.40455879e-02, 2.66378932e-03, 4.92295242e-03, 1.40776208e-03,
            1.38247294e-03, 1.01156557e-03, 5.73220488e-04, 1.41619179e-03]),
     'mask': array([False, False, False, False, False, False, False, False, False,
            False, False, False, False, False,  True, False,  True, False,
            False, False, False, False, False, False]),
     'target_replaced_anomaly': array([293.96      , 292.4215    , 290.7       , 289.79      ,
            288.88      , 288.23      , 287.82      , 287.13      ,
            286.44      , 286.27      , 285.96      , 285.65      ,
            285.15      , 283.61966667, 285.35      , 288.906     ,
            290.15      , 291.41      , 294.7       , 294.6225    ,
            295.76      , 296.24      , 296.57      , 296.11      ])}




    
![png](OBS_Notebook_files/OBS_Notebook_15_1.png)
    


## Optimal Baseline Subtraction Considerations
The OBS method described above is a simple yet powerful approach for detecting anomalies in time series data using historical patterns within a statistical dataset.
Let's list some considerations:
- This method can be considered as a **preprocessing approach** and can be done before applying Machine Learning methods. 
- This method is very versatile, and can be used as an **unsupervised approach**, as shown above. In presence of a labeled dataset (1/0 for anomaly/non anomaly), the choice of the threshold can be calibrated to achieve maximum accuracy
- The performance of this method increases when the historical dataset is **large**, as more optimal baseline (lower error) can be found. 
- The bank of candidates can be personalized, for example by considering only time series that happen **before** the target one, or only time series that belong to a certain class (e.g. only time series from the same city in the example above). 

## Application to Nixtla
Optimal Baseline Subtraction (OBS) can be seamlessly integrated into a forecasting pipeline using Nixtla’s ```StatsForecast``` library. After detecting and optionally replacing anomalies in your time series using OBS, you can feed the cleaned signal directly into models like AutoARIMA, AutoETS, or MSTL to improve forecasting accuracy. This is how you would do it:

### OBS as "Cleaner"
The OBS is here used as a "cleaner" for the time series: given your time series, you can replace the anomaly values of the target time series with the ones of the optimal baseline. For example, if the index 15 is an anomaly in the target time series, you replace the original value of the target time series with the one of the optimal baseline.


```python
from data import *
import pandas as pd
import numpy as np

# Step 1: Get target and baseline
data = TimeSeriesData()
data.isolate_city_and_time(city='Los Angeles', segment_class='day', segment_idx=377)
data.find_optimal_baseline()
threshold = 0.007
anomaly_data = data.run_anomaly_detection(threshold = threshold, plot = False)
fixed_target = anomaly_data['target_replaced_anomaly']
```

As we can see here, the OBS is fixing the areas where we have seen anomalies:


```python
plt.plot(data.optimal_baseline_data['target_curve'], label = 'Target Time Series', color ='k')
plt.plot(fixed_target, label = 'Target Time Series after OBS', color ='red')
plt.legend()
```




    <matplotlib.legend.Legend at 0x125b3cf40>




    
![png](OBS_Notebook_files/OBS_Notebook_20_1.png)
    


### Nixtla Forecast stage

Now we can add the Nixtla block and add the forecast block from Nixtla’s ```StatsForecast``` library, with this code: 


```python
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
timestamps = pd.date_range(start='2020-01-01', periods=len(fixed_target), freq='H')
df = pd.DataFrame({
    'ds': timestamps,
    'y': fixed_target,
    'unique_id': 'los_angeles'  # Required by StatsForecast
})
sf = StatsForecast(models=[AutoARIMA(season_length=10)], freq='H')
sf.fit(df)
forecast = sf.predict(h=24)
forecast.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>unique_id</th>
      <th>ds</th>
      <th>AutoARIMA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>los_angeles</td>
      <td>2020-01-02 00:00:00</td>
      <td>296.220402</td>
    </tr>
    <tr>
      <th>1</th>
      <td>los_angeles</td>
      <td>2020-01-02 01:00:00</td>
      <td>296.330804</td>
    </tr>
    <tr>
      <th>2</th>
      <td>los_angeles</td>
      <td>2020-01-02 02:00:00</td>
      <td>296.441205</td>
    </tr>
    <tr>
      <th>3</th>
      <td>los_angeles</td>
      <td>2020-01-02 03:00:00</td>
      <td>296.551607</td>
    </tr>
    <tr>
      <th>4</th>
      <td>los_angeles</td>
      <td>2020-01-02 04:00:00</td>
      <td>296.662009</td>
    </tr>
  </tbody>
</table>
</div>



The idea here is that we are doing the forecasting only on "clean" data, so that when we see clear differencies between our prediction and the target time series we can spot anomalous points efficiently. 


## Conclusions:

In this post, we explored how Optimal Baseline Subtraction (OBS) can be used as an intuitive statistical method for anomaly detection in time series. The idea is to compare the segment of interest of the time series (defined as **target time series**) to its most similar historical counterpart (defined as **optimal baseline**) through an error metrics (e.g. MAE). By considering the difference between target and optimal baseline, we identify anomalies using a fixed **threshold** value: every point that is above that threshold is considered an anomaly. Once identified, anomalies can be selectively corrected—allowing us to preserve the integrity of the original time series while removing noise that could interfere with forecasting.

OBS can be a preprocessing stage to be integrated with the Nixtla’s StatsForecast library. By forecasting on "clean" (i.e. anomaly-filtered time series), we reduce the risk of bias or distortion caused by outliers. This makes the forecast more reliable, and—crucially—any large discrepancies between forecasted and observed values can now be interpreted as new potential anomalies. Then, if we notice some differences between the predicted time series and the ground truth, we can easily spot the anomalies in future time steps.
