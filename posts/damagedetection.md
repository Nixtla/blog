---
title: "Damage Detection in Engineering Structures Using Nixtla"
description: "Learn how to detect cracks and structural damage using Nixtla's Anomaly Detection pipeline while accounting for temperature-induced variations in sensor data."
image: "/images/title_image.svg"
categories: ["TimeGPT Anomalies"]
tags:
  - TimeGPT
  - anomaly detection
  - structural health monitoring
  - damage detection
  - NDT
  - SHM
  - engineering structures
  - synthetic data
  - Python
author_name: Piero Paialunga
author_image: "/images/authors/piero.jpg"
author_position: Data Scientist
publication_date: 2025-01-12
---

# Introduction

Every bridge, building, and critical infrastructure around us is constantly under stress.
Wind loads, temperature fluctuations, traffic vibrations, and material aging all take their toll over time. The question that keeps structural engineers awake at night is simple yet profound: *How do we know when a structure is starting to fail before it becomes dangerous?*

This is where **Structural Health Monitoring (SHM)** comes in. SHM is a field that combines civil engineering, sensor technology, and data science to continuously monitor the integrity of structures. By installing networks of sensors (accelerometers, strain gauges, temperature sensors) on bridges, buildings, tunnels, and other critical infrastructure, engineers can collect time series data that captures the structure's "heartbeat."



As one can easily imagine, people who work in SHM deal with *a lot* of time series. Here is why time series are so much used in this field:

**1. Continuous Monitoring**: Unlike periodic inspections (checking a bridge once a year), time series captures every moment. A crack usually develops gradually, showing subtle changes in patterns over days, weeks, or months.

**2. Pattern Recognition**: Healthy structures have characteristic "signatures" in their sensor data. A bridge vibrates at certain frequencies when trucks pass over it. When damage like a crack appears, these patterns shift. Time series help us detect these shifts early.

**3. Environmental Context**: Here's the tricky part: structures naturally change with temperature. Steel expands in heat, concrete stiffness varies with weather, and thermal effects create their own patterns in the data. Time series let us track both the structure AND the environment, so we can separate normal temperature effects from actual damage.

**In this blog post, we'll tackle point 3: how to build a damage detection system that accounts for temperature-induced variations in sensor data.**

Here's our roadmap:

1. **Setting up a virtual experiment**: We'll create a realistic SHM scenario with simulated sensors on a structure
2. **Generating the data**: We'll build synthetic time series that capture temperature-dependent structural behavior
3. **Adding some anomalies**: We'll inject controlled "damage" signatures into our data to simulate cracks and structural issues
4. **Detecting them using Nixtla's pipeline**: We'll use Nixtla's anomaly detection tools to automatically identify damage while filtering out temperature effects

Let's get started.

# 1. The Scenario: Virtual Experiment Setup

Imagine an aircraft panel, a critical part of the fuselage that keeps passengers safe at 30,000 feet. Over time, tiny cracks can form from stress, vibration, or material fatigue. If left undetected, a small crack can grow into a catastrophic failure. **We need to catch these cracks early, before they become dangerous.**

In SHM, sensors send waves through the panel (imagine tapping a wall to check if it's hollow, but automated and happening thousands of times per second). In a healthy panel, these waves travel smoothly and create a predictable pattern when they bounce back. **But when a crack appears, it disrupts the wave pattern**: the waves scatter differently, arrive at different times, or have different amplitudes. It's like the structure's "signature" changes.

Our sensors record this signature as time series data, continuously tracking how the structure responds. By analyzing this data, we can detect the moment a crack starts forming, often long before it's visible to the naked eye.

Here's the problem: **temperature changes everything**. When it's cold (5°C), materials contract. When it's warm (25°C), they expand. This simple physical change affects our sensor readings: the waves travel at different speeds, have different amplitudes, and create different patterns. A signal that looks alarming at one temperature might be completely normal at another. Without accounting for temperature, we'd be constantly crying wolf.

Even worse, the relationship isn't always simple. Sometimes the signal amplitude changes linearly with temperature (thermal expansion). Other times it's polynomial (material stiffness shifts). And occasionally it's sinusoidal (thermal cycles creating periodic effects). In real SHM systems, you might see all three patterns depending on the operating conditions.

**This is exactly what we'll simulate**: sensor data where the temperature dependency randomly shifts between linear, polynomial, and sinusoidal relationships, mimicking the complex reality of structural monitoring.

# 2. Data Generation

## 2.1 Chirplet Wave

To simulate realistic SHM data, we need signals that look like what actual sensors measure. In the real world, when waves propagate through a structure, they create complex patterns. They have localized bursts of energy that decay over time. The mathematical name for these "wave packages" is **chirplet**. This is the function we will use:


```python
def chirplet(t, tau, fc, alpha1, alpha2=0.0, beta=1.0, phi=0.0):
    u = t - tau
    env = np.exp(-alpha1 * u**2)
    phase = 2*np.pi*(fc*u) + alpha2*u**2 + phi
    return beta * env * np.cos(phase)
```

And this is how it looks:



```python
# Generate simple chirplet
t = np.linspace(0, 1, 1000)
y = chirplet(t, tau=0.5, fc=10, alpha1=50)
plt.figure(figsize=(10, 6))
plt.plot(t, y, color='#98FE09', linewidth=2)
plt.title('Chirplet Signal', fontsize=16)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.3)
plt.show()
```


![Chirplet Signal](/images/chirplet.svg)

## 2.2 Full Signal

A single chirplet is just the building block. In reality, when you send a wave through a structure, it doesn't just travel in a straight line. It bounces off edges, scatters at damage sites, and takes multiple paths to reach the sensor. The result is a complex signal made up of many overlapping chirplets arriving at different times with different amplitudes. Without boring you with the details, the main components are: the primary burst (direct signal), echoes and reflections from structural features, and multiple path arrivals, each one contributing to the overall time series we observe. 

This is the function that we will use to model a signal:

```python

def generate_random_signal():
    rng = np.random.default_rng()
    fs, T = 1000, 0.6
    t = np.arange(int(T * fs)) / fs
    y = np.zeros_like(t)

    # small global jitter
    s = 1.0 + rng.normal(0, 0.0005)          # ~0.05% time stretch
    amp_scale = np.exp(rng.normal(0, 0.01))  # ~1% amplitude drift
    tt = t * s

    # primary burst
    y += chirplet(tt, tau=0.08, fc=120, alpha1=900, beta=1.0)

    # echoes (very small variation)
    echo_times = np.array([0.12, 0.16, 0.20, 0.26, 0.33, 0.41, 0.50])
    echo_gains = np.exp(-4.0*(echo_times-0.10)) * 0.6
    for tau0, g0 in zip(echo_times, echo_gains):
        tau   = tau0 + rng.normal(0, 4e-5)     # ~0.2 ms jitter
        fc    = 120 + rng.normal(0, 0.1)       # small freq jitter
        phi   = rng.uniform(-0.1, 0.1)         # small phase shift
        y += chirplet(tt, tau=tau, fc=fc, alpha1=800, beta=g0, phi=phi)

    # scattering coda (tiny jitter, smooth decay)
    for tau0 in np.linspace(0.22, 0.58, 100):
        tau = tau0 + rng.normal(0, 4e-5)
        fc    = 120 + rng.normal(0, 0.1)       # small freq jitter
        g   = 0.5 * np.exp(-6*(tau0-0.20))
        y  += chirplet(tt, tau=tau, fc=fc, alpha1=1200, beta=g)

    y *= amp_scale

    idx = pd.date_range(start="2025-01-01", periods=len(t), freq=f"{1000/fs}ms")
    df = pd.DataFrame({"unique_id": "sensor_A", "ds": idx, "y": y})
    return t, y, df
```

Let's visualize what this looks like:

```python
# Generate and plot a random signal
t, y, df = generate_random_signal()

plt.figure(figsize=(12, 4))
plt.plot(t, y, color='#98FE09', linewidth=1.5)
plt.title('Simulated Sensor Signal', fontsize=16)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.3)
plt.savefig('images/random_signal.svg')
plt.show()
```

![Random Signal](/images/random_signal.svg)

## 2.3 Full Dataset (Temperature Dependency)

The random signal we just generated is our starting point: it represents what a sensor might record at a single, fixed temperature. But in the real world, structures operate across a range of temperatures. An aircraft panel might see -20°C on the tarmac in winter and +40°C in summer. **We need to understand how our sensor signal changes across this entire temperature range.**

Here's what we'll do: imagine an engineer taking measurements at different temperatures in a climate-controlled lab. They set the temperature to 0°C, send a wave pulse, and record the signal. Then they increase it to 5°C, repeat the measurement, then 10°C, and so on up to 40°C. For each temperature, they get a slightly different signal amplitude and shape.

In our simulation, we'll do exactly this, but virtually. For every single time point in our 600-sample signal (that's 0.6 seconds of data at 1000 Hz), we'll create a mathematical relationship that describes how that particular moment's amplitude varies with temperature. 

There are three random options for the variation that we will simulate:

1. The relationship is **linear**, amplitude increases proportionally with temperature
2. The relationship is **polynomial**, a more complex, curved relationship between amplitude and temperature
3. The relationship is  **sinusoidal**, with cyclical variations

These different patterns, which are randomly injected for every index (out of the 600), mimic the complex physics happening in real materials as temperature changes.

The result? Instead of one signal at one temperature, we'll have a complete dataset showing how our sensor readings evolve across the entire temperature range: exactly what we need to build a robust damage detection system.

This is the code to do so:

```python
def chirplet(t, tau, fc, alpha1, alpha2=0.0, beta=1.0, phi=0.0):
    u = t - tau
    env = np.exp(-alpha1 * u**2)
    phase = 2*np.pi*(fc*u) + alpha2*u**2 + phi
    return beta * env * np.cos(phase)


def generate_mixed_dependency_timeseries(n_temperatures=100, temperature_range=(0, 1), 
                                        fs=1000, T=0.6, seed=42, change_probability=0.1):

    rng = np.random.default_rng(seed)
    
    # Generate time array
    t = np.arange(int(T * fs)) / fs
    k = len(t)  # number of time steps
    
    # Generate temperature values
    temperature_values = np.linspace(temperature_range[0], temperature_range[1], n_temperatures)
    
    # Initialize arrays
    timeseries_data = np.zeros((k, n_temperatures))
    dependency_types = np.zeros(k, dtype=int)  # 0=linear, 1=polynomial, 2=sinusoidal
    dependency_names = ['linear', 'polynomial', 'sinusoidal']
    
    # Generate column names
    column_names = [f"temp_{temp:.3f}" for temp in temperature_values]
    
    # Generate base signal for reference
    base_t, base_y, base_df = generate_random_signal()
    
    # Start with random dependency type
    current_dependency = rng.choice([0, 1, 2])
    dependency_types[0] = current_dependency
    
    
    # For each time step, create a mathematical relationship across temperatures
    for time_idx in range(k):
        base_value = base_y[time_idx]
        
        # Check if we should change dependency type
        if time_idx > 0 and rng.random() < change_probability:
            current_dependency = rng.choice([0, 1, 2])
            print(f"Time step {time_idx}: {dependency_names[current_dependency]}")
        
        dependency_types[time_idx] = current_dependency
        
        if current_dependency == 0:  # Linear
            # Linear dependency: signal = a * temp + b
            a = rng.uniform(-0.05, 0.05)  # Random slope between 0.01 and 0.05
            b = base_value  # intercept
            for temp_idx, temp in enumerate(temperature_values):
                timeseries_data[time_idx, temp_idx] = a * temp + b
                
        elif current_dependency == 1:  # Polynomial
            # Polynomial dependency: signal = a * temp^2 + b * temp + c
            # Random coefficients within reasonable ranges
            a = rng.uniform(-0.005, 0.02)  # Random quadratic coefficient
            b = rng.uniform(0.01, 0.03)   # Random linear coefficient
            c = base_value  # constant
            for temp_idx, temp in enumerate(temperature_values):
                timeseries_data[time_idx, temp_idx] = a * temp**2 + b * temp + c
                
        elif current_dependency == 2:  # Sinusoidal
            # Sinusoidal dependency: signal = amplitude * sin(freq * temp + phase) + offset
            amplitude = rng.uniform(-0.1, 0.1)  # Random amplitude
            freq = rng.uniform(1, 4) * np.pi  # Random frequency (1π to 4π)
            phase = rng.uniform(0, 2*np.pi)  # Random phase
            offset = base_value  # offset
            for temp_idx, temp in enumerate(temperature_values):
                timeseries_data[time_idx, temp_idx] = amplitude * np.sin(freq * temp + phase) + offset
    
    # Add small amount of noise for realism
    noise_std = 0.0002  # Small but noticeable noise
    for time_idx in range(k):
        noise = rng.normal(0, noise_std, n_temperatures)
        timeseries_data[time_idx, :] += noise
    
    return t, timeseries_data, temperature_values, column_names, dependency_types
```

This will create a `timeseries_size x num_temperatures` matrix. As our time series has 600 points and we are using `num_temperatures = 100`, by default we will have a `600 x 100` matrix. Let's plot some columns.

```python
def plot_random_temperatures(timeseries_data, temperature_values, n_temps=6, seed=42):
    """Plot n random temperature time series."""
    np.random.seed(seed)
    temp_indices = np.random.choice(len(temperature_values), n_temps, replace=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    colors = ['#02FEFA']*6
    
    for i, temp_idx in enumerate(temp_indices):
        ax = axes[i]
        ax.plot(timeseries_data[:, temp_idx], color=colors[i], linewidth=2)
        ax.set_title(f'Temperature = {temperature_values[temp_idx]:.3f}', fontsize=14)
        ax.set_xlabel('Time Steps', fontsize=15)
        ax.set_ylabel('Signal Value', fontsize=15)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

![Random Signal with temperatures](/images/random_temperatures.svg)

Now the question to answer is the following:

*At what index, and what temperature, does the signal matrix present an anomaly?*

From the following plot, it is extremely hard to answer this question, but if we transpose the signal matrix and plot for a fixed index **at different temperatures**, we get the following:

```python
time_steps_to_show = [0, 10, 22]
dependency_names = ['linear', 'polynomial', 'sinusoidal']
dep_colors = {'linear': '#98FE09', 'polynomial': '#02FEFA', 'sinusoidal': '#0E00F8'}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes = axes.flatten()
for i, time_step in enumerate(time_steps_to_show):
    ax = axes[i]
    dep_type = dependency_types[time_step]
    dep_name = dependency_names[dep_type]
    color = dep_colors[dep_name]
    
    ax.plot(temperature_values, timeseries_mixed[time_step, :], 'o-', 
           markersize=2, color=color, alpha=0.8)
    ax.set_xlabel('Temperature', fontsize=12)
    ax.tick_params(axis='x', labelsize=14)  
    ax.tick_params(axis='y', labelsize=14)  
    ax.set_xlabel('Temperature', fontsize=14)  
    ax.set_ylabel('Time Series Value', fontsize=14)  

    ax.set_title(f'Time Step {time_step} ({dep_name})', fontsize=20)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```
![Dependency Types](/images/dependency_types.svg)


Now, if at temperature = 1 you see -0.06 at time step = 0, then it is obviously not an anomaly, because it follows the linear trend. However, if you see the same value at time step = 22, there is obviously a problem. 

This approach is crucial because in real-world monitoring, we can't control the temperature. A crack might form when the aircraft is at 15°C, but we're trying to detect it using measurements taken at 25°C. Without understanding the temperature dependency, we'd have no way to distinguish "the signal changed because of temperature" from "the signal changed because of damage." Our synthetic dataset, with its known temperature relationships, gives us the perfect test set to develop and validate detection algorithms that can make this distinction reliably.

Perfect, so let's see how these anomalies will look like. 

# 3. Anomaly Creation

Now comes the interesting part: **injecting damage into our synthetic data**. We have a clean dataset showing normal structural behavior across different temperatures. To test our detection algorithms, we need to add controlled "anomalies". 

In reality, damage/anomalies can manifest in many ways: gradual signal drift over time, sudden level shifts, or complex pattern changes. However, one of the most common and detectable signatures is a **spike anomaly**, a sudden, localized jump in amplitude at a specific time and temperature. This mimics what happens when a crack causes an unexpected wave reflection or scattering event.

For simplicity, we'll focus on spike anomalies in this tutorial. The concept is straightforward: we pick a specific time step (say, index 30 out of our 600 samples) and a specific temperature (say, 0.5), and we add a sudden amplitude increase at that exact point in our data matrix. This creates a clear, detectable deviation from the expected temperature-dependent pattern. 

This is the function we will use:

```python

def add_anomaly(timeseries_data, time_step, temperature_value, spike_magnitude=0.1, temperature_values=None):
    data = timeseries_data.copy()
    
    if temperature_values is not None:
        # Find closest temperature index
        temp_idx = np.argmin(np.abs(temperature_values - temperature_value))
        actual_temp = temperature_values[temp_idx]
    else:
        # Assume temperature_value is already an index
        temp_idx = int(temperature_value * len(timeseries_data[0]))
        actual_temp = temperature_value
    
    # Add spike anomaly
    data[time_step, temp_idx] += spike_magnitude
    
    print(f"Added spike anomaly:")
    print(f"  Time step: {time_step}")
    print(f"  Temperature: {actual_temp:.3f} (index {temp_idx})")
    print(f"  Spike magnitude: {spike_magnitude}")
    
    return data

```

Let's see how this looks:

1. Creation of the dataset:

```python
t_mixed, timeseries_mixed, temperature_values, column_names, dependency_types = generate_mixed_dependency_timeseries(
    n_temperatures=100, 
    temperature_range=(0, 1), 
    fs=1000,
    T=0.6,
    seed=42,
    change_probability=0.1  # 10% chance of changing dependency type at each time step
)
```
2. Adding the anomaly: 
```python
# Example: Add anomaly at time step 30, temperature 0.5
timeseries_with_anomaly = add_anomaly(
    timeseries_mixed, 
    time_step=30, 
    temperature_value=0.5, 
    spike_magnitude=0.01,
    temperature_values=temperature_values
)
```

3. Displaying the anomaly:
``` python

plt.figure(figsize=(12, 6))
time_step = 30
plt.plot(temperature_values, timeseries_mixed[time_step, :], 'o-', color='#98FE09', linewidth=2, label='Original')
plt.plot(temperature_values, timeseries_with_anomaly[time_step, :], 'o-', color='cyan', linewidth=2, label='With Anomaly')
plt.axvline(x=0.5, color='lime', linestyle='--', alpha=0.7, label='Anomaly Location')
plt.title(f'Time Step {time_step} - Signal vs Temperature (Before/After Anomaly)', fontsize=16)
plt.xlabel('Temperature')
plt.ylabel('Signal Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('images/anomaly_detection.svg')
plt.show()
```

![Dependency Types](/images/anomaly_detection_input.svg)


Now that we have injected the anomaly, let's see if we are going to be able to detect it. 

# 4. Anomaly Detection through Nixtla

### 4.1 Anomaly Detection Algorithm

The detection approach leverages **Nixtla's StatsForecast** library with an **AutoETS (Exponential Smoothing State Space)** model. The core idea is elegant: instead of trying to manually define what "normal" looks like at each temperature, we let a probabilistic forecasting model learn the expected pattern from the data itself. The model analyzes the time series across temperatures, fits to the underlying trends and relationships, and produces **prediction intervals**, essentially confidence bands that represent where we expect normal values to fall.

Here's the process for a given time step:

1. Feed the signal values across all temperatures into the AutoETS model
2. The model generates fitted values along with upper and lower prediction bounds (typically at 90% confidence)
3. Any data point falling **outside these prediction intervals** is flagged as an anomaly

This approach automatically accounts for complex temperature dependencies, and only raises "alarms" when values truly deviate from expected behavior.

### 4.2 Algorithm implementation

It might sound complicated, but the implementation is extremely simple. 

```python
from statsforecast import StatsForecast
from statsforecast.models import AutoETS   # simple, fast probabilistic model
# You could use AutoARIMA/MSTL/etc. too.

def detect_anomalies(
    timeseries_data: np.ndarray,
    temperature_values: pd.DatetimeIndex, timegpt_level: float = 0.90
) -> pd.DataFrame:
    y = np.asarray(timeseries_data, dtype=float)
    n = len(y)
    if len(temperature_values) != n:
        raise ValueError("temperature_values length must match timeseries_data length.")

    # Build long-format df expected by StatsForecast
    start = pd.Timestamp("2024-01-01")
    ds = pd.date_range(start, periods=n, freq="D")
    df = pd.DataFrame({"unique_id": 0, "ds": ds, "y": y})

    # Model & forecaster
    # AutoETS provides probabilistic insample/fitted intervals easily.
    levels = [int(round(timegpt_level * 100))]  # e.g., 90 -> 90% PI
    sf = StatsForecast(models=[AutoETS(season_length=1)], freq="D", n_jobs=-1)

    # Fit & get forecasts with fitted=True so we can retrieve INSAMPLE intervals
    _ = sf.forecast(df=df, h=1, level=levels, fitted=True)

    # Retrieve insample fitted values & intervals
    insample = sf.forecast_fitted_values()  # columns: unique_id, ds, y, <model>, <model>-lo-XX, <model>-hi-XX
    model_name = [c for c in insample.columns if c not in ("unique_id", "ds", "y") and "-lo-" not in c and "-hi-" not in c][0]
    lo_col = f"{model_name}-lo-{levels[0]}"
    hi_col = f"{model_name}-hi-{levels[0]}"

    # Flag anomalies: outside the interval
    is_anom = ~insample["y"].between(insample[lo_col], insample[hi_col])
    is_anom = is_anom.astype(int).to_numpy()

    # Assemble output to match your structure
    out = pd.DataFrame({
        "ds": insample["ds"].values,
        "y": insample["y"].values,
        "unique_id": 0,
        "is_anomaly": is_anom,
        "temperature_values": temperature_values,
    })
    # Optional: stash columns for plotting convenience
    out["_fitted_mean"] = insample[model_name].values
    out["_lo"] = insample[lo_col].values
    out["_hi"] = insample[hi_col].values
    out.attrs["model_name"] = model_name
    out.attrs["level"] = levels[0]
    return out



df_out = detect_anomalies( timeseries_data=timeseries_with_anomaly[time_step,:], temperature_values=temperature_values)
```

And we can see from the following plot that we are able to retrieve the anomaly very well!

```python

plt.plot(df_out['temperature_values'].values,df_out['y'].values, color='#98FE09', linewidth=2, label='Original')
plt.plot(df_out[df_out['is_anomaly']==1]['temperature_values'].values,df_out[df_out['is_anomaly']==1]['y'].values, '.', color='#02FEFA', linewidth=5, label='Anomaly', markersize=10)
plt.xlabel('Temperature')
plt.ylabel('Signal Value')
plt.title(f'Anomaly Detection at time step : {time_step}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

![Detected Anomalies](/images/anomaly_detection.svg)

There we go: **anomaly detected**! 

From a time series perspective, we can see the following plot. 

```python
temperature_idx = np.argmin(np.abs(temperature_values-0.5))
plt.plot(t_mixed, timeseries_with_anomaly[:,temperature_idx], color='#98FE09', linewidth=2, label='Original')
plt.axvline(x=t_mixed[30], color='#02FEFA', linestyle='--', alpha=0.7, label='Anomaly Time Step')
plt.xlabel('Time')
plt.ylabel('Signal Value')
plt.title(f'Anomaly Detection at time step : {time_step}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```
![Detected Anomalies Time Series](/images/anomaly_detection_ts.svg)


As we can see, that point doesn't look like an anomaly *per se* but it is clearly an anomaly when we consider the temperature perspective: well done!

# Conclusion

Let's recap what we covered in this post:

- **We introduced Structural Health Monitoring (SHM)** and explained why continuous sensor monitoring is critical for detecting cracks in structures like aircraft panels.

- **We built a synthetic structural monitoring dataset** that mimics real-world dynamics: chirplet-based sensor signals with complex temperature dependencies (linear, polynomial, and sinusoidal relationships). This approach captures the physics of how structures respond to environmental changes.

- **We explicitly injected spike anomalies** to simulate structural damage like cracks. Since we know exactly where and when we added these anomalies, we can verify that our detection algorithm correctly identifies them at the right time steps and temperatures.

- **We demonstrated temperature-compensated anomaly detection**. By using Nixtla's StatsForecast with AutoETS, we built a system that learns normal temperature-dependent behavior and only flags true deviations (not just temperature effects).

- **We validated the approach**. The algorithm successfully captured the complex temperature patterns across our dataset while reliably detecting the injected spike anomaly at time step 30, temperature 0.5, which is exactly where we placed it.

Overall, this workflow shows how synthetic data, combined with Nixtla's forecasting models, provides a robust foundation for damage detection in engineering structures. 


