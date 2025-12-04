# Chart Data Structure Documentation

## Chart Types

This documentation covers two chart configurations:

1. **Single Chart** - One visualization with multiple series
2. **Multiple Charts** - Grid layout with multiple synchronized charts

---

## Single Chart Configuration

### Top-Level Configuration

| Field        | Type                 | Required | What it does                                        | Example                                       |
| ------------ | -------------------- | -------- | --------------------------------------------------- | --------------------------------------------- |
| `id`         | `string`             | Yes      | Unique identifier for the chart                     | `"temperature-signal"`                        |
| `title`      | `string`             | Yes      | Chart title displayed above visualization           | `"Daily Website Traffic"`                     |
| `dataSource` | `string`             | Yes      | CSV filename (located in `posts/{post-name}/data/`) | `"temperature-signal.csv"`                    |
| `xAxis`      | `string` or `object` | Yes      | X-axis configuration                                | `"date"` or `{"key": "date"}`                 |
| `yAxis`      | `string` or `object` | Yes      | Y-axis configuration                                | `"Revenue ($)"` or `{"label": "Revenue ($)"}` |
| `series`     | `array`              | Yes      | Array of series objects defining what to plot       | See Series Configuration below                |
| `anomalies`  | `object`             | No       | Configuration for anomaly markers                   | See Anomalies Configuration below             |
| `thresholds` | `object`             | No       | Configuration for vertical threshold lines          | See Thresholds Configuration below            |

### Series Configuration

For Line and Bar Charts

| Field             | Type      | Required | What it does                                              | Example                         |
| ----------------- | --------- | -------- | --------------------------------------------------------- | ------------------------------- |
| `column`          | `string`  | Yes      | Which CSV column to plot                                  | `"visits"`, `"temperature"`     |
| `type`            | `string`  | Yes      | How to draw it                                            | `"line"`, `"bar"`               |
| `name`            | `string`  | No       | Label in the legend (defaults to column name)             | `"Daily Views"`                 |
| `color`           | `string`  | No       | Color from design system or any tailwind css color tokens | `"chart-1"`, `"rose-500"`, etc. |
| `strokeWidth`     | `number`  | No       | Line thickness for line charts ( default is 2)            | `3`, `5`                        |
| `showDots`        | `boolean` | No       | Show dots on data points for line charts                  | `true`, `false`                 |
| `strokeDashArray` | `string`  | No       | Pattern for dashed lines                                  | `"5 5"`, `"10 5"`               |

#### For Area Charts (Confidence Intervals)

| Field          | Type     | Required | What it does                                             | Example                                      |
| -------------- | -------- | -------- | -------------------------------------------------------- | -------------------------------------------- |
| `type`         | `string` | Yes      | Must be `"area"`                                         | `"area"`                                     |
| `columns`      | `object` | Yes      | High and low boundary columns                            | `{"high": "conf95High", "low": "conf95Low"}` |
| `columns.high` | `string` | Yes      | CSV column for upper boundary                            | `"interval95High"`                           |
| `columns.low`  | `string` | Yes      | CSV column for lower boundary                            | `"interval95Low"`                            |
| `name`         | `string` | No       | Label in the legend                                      | `"95% Confidence"`                           |
| `color`        | `string` | No       | Color from design system or any tailwind css color token | `"chart-2"`, `cyan-400`                      |

### Anomalies Configuration

Top-level `anomalies` object:

| Field          | Type      | Required | What it does                                                                                            | Example                             |
| -------------- | --------- | -------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| `enabled`      | `boolean` | Yes      | Whether to show anomaly markers                                                                         | `true`, `false`                     |
| `column`       | `string`  | Yes      | CSV column with boolean anomaly flags (commonly: `"anomaly"` or `"is_anomaly"`)                         | `"anomaly"`, `"is_anomaly"`         |
| `seriesColumn` | `string`  | Yes      | Which data column anomalies apply to                                                                    | `"visits"`, `"y"`                   |
| `color`        | `string`  | No       | Color for anomaly markers or any tailwind css color tokens (defaults is `"lime-500"` from tailwind css) | `"chart-4"`, `purple-500`           |
| `marker`       | `string`  | No       | Shape of anomaly marker (defaults to `"cross"`)                                                         | `"cross"`, `"circle"`, `"triangle"` |
| `size`         | `number`  | No       | Size of anomaly marker (defaults to `4`)                                                                | `3`, `5`, `10`                      |
| `label`        | `string`  | No       | Label for legend (defaults to `"Anomaly Detected"`)                                                     | `"Anomaly Detected"`                |

**Note:** The system automatically recognizes boolean columns named `"anomaly"`, `"is_anomaly"`, or `"threshold"` in your CSV data.

### Thresholds Configuration

Top-level `thresholds` object:

| Field             | Type      | Required | What it does                                                                         | Example               |
| ----------------- | --------- | -------- | ------------------------------------------------------------------------------------ | --------------------- |
| `enabled`         | `boolean` | Yes      | Whether to show threshold lines                                                      | `true`, `false`       |
| `column`          | `string`  | Yes      | CSV column with boolean threshold flags (commonly: `"threshold"`)                    | `"threshold"`         |
| `label`           | `string`  | No       | Label for legend (defaults to `"Threshold"`)                                         | `"Anomaly Time Step"` |
| `color`           | `string`  | No       | Color for threshold lines or any tailwind css color tokens (defaults to `"chart-1"`) | `"chart-3"`           |
| `strokeDashArray` | `string`  | No       | Pattern for dashed lines (defaults to `"5 5"`)                                       | `"5 5"`, `"10 5"`     |
| `strokeWidth`     | `number`  | No       | Width of threshold lines (defaults to `2`)                                           | `2`, `3`              |

**Note:** The system automatically recognizes boolean columns named `"anomaly"`, `"is_anomaly"`, or `"threshold"` in your CSV data.

---

## Multiple Charts Configuration

For displaying multiple synchronized charts in a grid layout, sharing the same data source and legend.

### Top-Level Configuration

| Field          | Type                 | Required | What it does                                                          | Example                                     |
| -------------- | -------------------- | -------- | --------------------------------------------------------------------- | ------------------------------------------- |
| `id`           | `string`             | Yes      | Unique identifier for the chart group                                 | `"chart-multiple-1"`                        |
| `title`        | `string`             | Yes      | Main title displayed above all charts                                 | `"Selected Series"`                         |
| `dataSource`   | `string`             | Yes      | CSV filename shared by all charts                                     | `"chart-1.csv"`                             |
| `columns`      | `number`             | Yes      | Number of columns in the grid layout                                  | `2`, `3`, `4`                               |
| `legendConfig` | `object`             | Yes      | Shared legend configuration for all charts                            | `{"displaySeries": [...]}`                  |
| `maxPoints`    | `number`             | No       | Maximum number of points to load from the CSV file                    | `60 (default is 1000)`                      |
| `xAxis`        | `string` or `object` | Yes      | X-axis configuration (shared by all charts)                           | `"date"` or `{"key": "date"}`               |
| `yAxis`        | `string` or `object` | Yes      | Y-axis configuration (shared by all charts)                           | `"Target (y)"` or `{"label": "Target (y)"}` |
| `charts`       | `array`              | Yes      | Array of individual chart configurations (each follows series schema) | See Charts Array Configuration below        |

### Legend Configuration

The `legendConfig` object defines the shared legend displayed above all charts:

| Field           | Type    | Required | What it does                                      | Example                                                                         |
| --------------- | ------- | -------- | ------------------------------------------------- | ------------------------------------------------------------------------------- |
| `displaySeries` | `array` | Yes      | Array of series definitions for the shared legend | `[{"name": "Y", "color": "blue-500"}, {"name": "Actual", "color": "cyan-500"}]` |

Each item in `displaySeries`:

| Field   | Type     | Required | What it does                                     | Example                    |
| ------- | -------- | -------- | ------------------------------------------------ | -------------------------- |
| `name`  | `string` | Yes      | Display name in legend                           | `"Y"`, `"Actual"`          |
| `color` | `string` | Yes      | Color for this series (must match series colors) | `"blue-500"`, `"cyan-500"` |

### Charts Array Configuration

Each object in the `charts` array represents one chart in the grid:

| Field    | Type     | Required | What it does                                                                    | Example           |
| -------- | -------- | -------- | ------------------------------------------------------------------------------- | ----------------- |
| `id`     | `string` | Yes      | Unique identifier for this individual chart                                     | `"chart-inner-1"` |
| `series` | `array`  | Yes      | Array of series for this chart (follows Series Configuration from Single Chart) | See example below |

The `series` array follows the same schema as Single Chart series configuration.

---

## Complete Examples

### Simple Line Chart

```json
{
  "id": "temperature-signal",
  "title": "Temperature Over Time",
  "dataSource": "temperature-signal.csv",
  "xAxis": "timeStep",
  "yAxis": "Temperature (°C)",
  "series": [
    {
      "column": "temperature",
      "type": "line",
      "name": "Temperature"
    }
  ]
}
```

### Line Chart with Anomalies

```json
{
  "id": "daily-traffic-anomalies",
  "title": "Daily Website Traffic with Anomalies",
  "dataSource": "daily-traffic.csv",
  "xAxis": "date",
  "yAxis": "Number of Visits",
  "series": [
    {
      "column": "visits",
      "type": "line",
      "name": "Daily Traffic"
    }
  ],
  "anomalies": {
    "enabled": true,
    "column": "anomaly",
    "seriesColumn": "visits",
    "color": "teal-500",
    "marker": "cross",
    "size": 2
  }
}
```

### Area Chart with Confidence Intervals

```json
{
  "id": "sales-forecast",
  "title": "Sales Forecast",
  "dataSource": "sales-forecast.csv",
  "xAxis": "date",
  "yAxis": "Revenue ($)",
  "series": [
    {
      "type": "area",
      "columns": {
        "high": "conf95High",
        "low": "conf95Low"
      },
      "name": "95% Confidence"
    },
    {
      "column": "actual",
      "type": "line",
      "name": "Actual Revenue"
    }
  ]
}
```

### Chart with Thresholds

```json
{
  "id": "performance-metrics",
  "title": "Performance Metrics",
  "dataSource": "metrics.csv",
  "xAxis": "timestamp",
  "yAxis": "Response Time (ms)",
  "series": [
    {
      "column": "responseTime",
      "type": "line",
      "name": "Response Time"
    }
  ],
  "thresholds": {
    "enabled": true,
    "column": "isThreshold",
    "label": "SLA Limit",
    "color": "chart-3",
    "strokeDashArray": "5 5"
  }
}
```

### Multiple Charts in Grid Layout

```json
{
  "id": "chart-multiple-1",
  "title": "Selected Series",
  "dataSource": "chart-1.csv",
  "columns": 2,
  "legendConfig": {
    "displaySeries": [
      { "name": "Y", "color": "blue-500" },
      { "name": "Actual", "color": "cyan-500" }
    ]
  },
  "xAxis": { "key": "ds" },
  "yAxis": { "label": "Target (y)" },
  "charts": [
    {
      "id": "chart-inner-1",
      "series": [
        {
          "column": "y_H51",
          "name": "y_H51",
          "type": "line",
          "color": "blue-500",
          "strokeWidth": 1
        },
        {
          "column": "actual_H51",
          "name": "actual_H51",
          "type": "line",
          "color": "cyan-500",
          "strokeWidth": 1
        }
      ]
    },
    {
      "id": "chart-inner-2",
      "series": [
        {
          "column": "y_H263",
          "name": "y_H263",
          "type": "line",
          "color": "blue-500",
          "strokeWidth": 1
        },
        {
          "column": "actual_H263",
          "name": "actual_H263",
          "type": "line",
          "color": "cyan-500",
          "strokeWidth": 1
        }
      ]
    },
    {
      "id": "chart-inner-3",
      "series": [
        {
          "column": "y_H25",
          "name": "y_H25",
          "type": "line",
          "color": "blue-500",
          "strokeWidth": 1
        },
        {
          "column": "actual_H25",
          "name": "actual_H25",
          "type": "line",
          "color": "cyan-500",
          "strokeWidth": 1
        }
      ]
    },
    {
      "id": "chart-inner-4",
      "series": [
        {
          "column": "y_H69",
          "name": "y_H69",
          "type": "line",
          "color": "blue-500",
          "strokeWidth": 1
        },
        {
          "column": "actual_H69",
          "name": "actual_H69",
          "type": "line",
          "color": "cyan-500",
          "strokeWidth": 1
        }
      ]
    }
  ]
}
```

---

## Notes

- **Multiple Charts**: All charts in the grid share the same `dataSource`, `xAxis`, `yAxis`, and legend configuration
- **Legend Colors**: Colors specified in `legendConfig.displaySeries` must match the colors used in individual chart series
- **Grid Layout**: The `columns` property controls how many charts appear per row. Charts wrap to new rows automatically
- **Synchronization**: All charts in a multiple chart configuration are visually synchronized with consistent styling and scales
