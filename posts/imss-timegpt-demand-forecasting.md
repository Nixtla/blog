---
title: "IMSS Scales Nationwide Demand Forecasting for Medicines and Medical Supplies With TimeGPT"
seo_title: IMSS Scales Nationwide Demand Forecasting With TimeGPT
description: "Learn how Mexico's largest public health institution uses TimeGPT to generate forecasts for over two million time series in under five minutes, achieving 15–18% improvement in forecast accuracy."
image: "/images/imss-timegpt-demand-forecasting/cover.png"
categories: ["Case Study"]
tags:
  - TimeGPT
  - forecasting
  - healthcare
  - case study
  - time series
author_name: Han Wang
author_image: "/images/authors/han.jpeg"
author_position: CTO - Nixtla
publication_date: 2026-05-20
---

The Instituto Mexicano del Seguro Social (IMSS) implemented TimeGPT to support large-scale, reliable demand forecasting for medicines and medical supplies across Mexico's public healthcare system. The solution enables IMSS to generate fast, consistent, and reproducible forecasts at a national scale, supporting multi-year planning, procurement, and budget allocation decisions.

With TimeGPT in production, IMSS is able to:

- Generate forecasts for more than two million time series in less than five minutes.
- Achieve 15–18% improvement in forecast accuracy (MAE) compared to the previous solution across all levels of granularity.
- Reduce implementation time from months to days compared to prior approaches and alternative solutions considered.
- Produce multi-year forecasts covering planning horizons from 2026 to 2028.
- Estimate demand for new products with no historical usage.

## About IMSS

The Instituto Mexicano del Seguro Social (IMSS) is Mexico's largest public health and social security institution, providing healthcare services to millions of beneficiaries nationwide. IMSS is responsible for the procurement, distribution, and management of medicines and medical supplies across a complex national network that includes regional administrative bodies (OOADs) and thousands of medical units, including hospitals and clinics.

Accurate demand forecasting is a critical input for IMSS operations. Forecasts directly inform procurement volumes, inventory management, and budget planning, helping ensure the availability of essential medicines while avoiding excess inventory and inefficient use of public funds. Given the scale and complexity of its mandate, IMSS requires forecasting systems that are transparent, scalable, and reliable.

## Challenge

### Forecasting demand at the national scale with flexibility

Prior to adopting TimeGPT, IMSS relied primarily on traditional econometric models and spreadsheet-based workflows to forecast demand for medicines and medical supplies. These processes required significant manual intervention by specialized personnel and were difficult to scale as forecasting needs increased in volume, granularity, and planning horizon.

Forecasts were typically produced at an aggregate product level, limiting visibility into regional and medical unit-level differences in demand. Updating forecasts in response to data corrections, late reporting, or policy changes required extensive manual rework, reducing the organization's ability to adapt quickly. The existing process also did not support reliable demand estimation for new products with no historical usage, a recurring requirement in public-sector procurement.

In addition, IMSS must produce consistent forecasts over long planning horizons to support multi-year budgeting and procurement cycles. Generating granular, long-horizon forecasts across millions of time series placed a heavy operational burden on teams and limited flexibility, increasing risk in inventory planning and resource allocation.

## Solution

### TimeGPT enables scalable, zero-shot forecasting across products, regions, and medical units

IMSS adopted TimeGPT as a unified forecasting engine for medicines and medical supplies. TimeGPT is a pre-trained foundation model for time series forecasting that enables zero-shot forecasting, allowing it to generate accurate forecasts without task-specific model training or manual parameter tuning.

Using TimeGPT, IMSS generates monthly demand forecasts at multiple levels of granularity, including the product level, the product and OOAD level, and the product and medical unit level. In total, the forecasting pipeline covers more than 3,500 product-level series, over 100,000 product and OOAD series, and more than two million product and medical unit series. Despite this scale, forecasts can be generated in under five minutes, enabling IMSS to update projections quickly whenever historical data is revised or planning assumptions change.

TimeGPT also enables demand estimation for new products with no historical usage by leveraging learned patterns from similar products, categories, and regional contexts. This capability supports procurement planning when products are introduced, replaced, or reformulated. Importantly, the system supports long-horizon forecasting, allowing IMSS to generate monthly demand projections covering multiple years. These forecasts align with IMSS's budgeting and procurement cycles and support consistent planning across regions and medical units.

## Impact

### Faster, more flexible demand planning at a national scale

With TimeGPT, IMSS transformed its demand forecasting process from a labor-intensive and rigid workflow into a fast, scalable, and easily reproducible system. The results are significant across multiple dimensions.

In terms of time to produce final, decision-ready forecasts, processes that previously required weeks of manual work and repeated iterations can now be completed in under five minutes, even at the most granular level of medical units. This reduction is driven by the elimination of manual forecasting steps, enabling IMSS to update projections rapidly in response to data corrections or revised assumptions.

Implementation was also significantly faster. Deploying TimeGPT took a single day, compared to the months typically required to build and validate alternative forecasting systems and solutions considered by IMSS.

Forecast accuracy improved by 15–18% in MAE compared to the previous solution, with gains observed across all levels of granularity: 18% at the product level, 15% at the product and OOAD level, and 16% at the product and medical unit level.

The ability to forecast millions of time series efficiently allows IMSS to plan inventory and budgets with greater confidence, while maintaining consistency across products, regions, and medical units. Support for new products and long-term horizons improves readiness for future procurement cycles and policy changes, while the replicability of the forecasting pipeline strengthens transparency and governance.

By adopting a foundation model-based approach to time series forecasting, which represents the state of the art in the field, IMSS gains a robust analytical backbone for nationwide healthcare planning that scales with the institution's operational complexity and evolving needs.
