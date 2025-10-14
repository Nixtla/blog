---
title: "TimeGPT 2: The Next Generation of Foundation Models for Time Series Forecasting"
description: "Replace hours of custom feature engineering code with MLforecast's automated lag features, rolling statistics, and target transformations for faster, more reliable time series forecasting."
categories: ["Time Series Forecasting"]
tags:
  - MLforecast
  - automated feature engineering
  - lag features
  - target transformations
image: "/images/automated-time-series-feature-engineering-with-mlforecast/automated-feature-engineering-rolling-expanding-comparison.svg"
author_name: Khuyen Tran
author_image: "/images/authors/khuyen.jpeg"
author_position: Developer Advocate - Nixtla
publication_date: 2025-08-26
---

At Nixtla, we're committed to pushing the boundaries of time series forecasting and making state-of-the-art algorithms accessible to everyone. With **TimeGPT-1**, we proved for the first time that pre-trained foundation models were possible for time series. Now, we're excited to introduce **TimeGPT-2**, a modular, production-ready model family that sets a new standard in accuracy and scalability.

## Modular and Purpose-Built

**TimeGPT 2 is not a single model, but a modular family designed for real world forecasting.**

Each variant is built on a shared foundation and fine tuned for different accuracy latency tradeoffs and hardware environments, enabling flexible deployment across industries and scales.

We are introducing **three models**, each designed to meet specific operational requirements:

- `timegpt-2-mini` – ideal for fast inference on resource constrained environments.
- `timegpt-2` – the best balance between compute cost and accuracy.
- `timegpt-2-pro` – the flagship model offering maximum accuracy. It surpasses TimeGPT 1.0 by more than 60% across short and long horizons and multiple frequencies.

This modular approach allows organizations to choose the model that best fits their operational constraints, without sacrificing accuracy or ease of deployment.

## Unmatched Accuracy

TimeGPT-2 sets a new benchmark for performance, **outperforming both foundation and classical models**. Its accuracy holds consistently across domains, frequencies, and applications, demonstrated through real-world use cases, internal evaluations, and public benchmarks, including **GIFT-EVAL** and **FEV-Bench**.

- `timegpt-2-pro` achieves #1 MASE in Gift Eval (0.7021) and 2nd place skill score in FEV (0.455). In Nixtla’s internal benchmarks, it surpasses TimeGPT 1.0 by more than 60% across short and long horizons and multiple frequencies.
- `timegpt-2` delivers top performance with a MASE of 0.715 in Gift Eval and a skill score of 0.430 in FEV.
- `timegpt-2-mini` provides lightning fast inference while maintaining competitive accuracy.

## Enterprise-Ready for Production

TimeGPT-2 is designed to deploy seamlessly into production.

- Deploy on-premise in minutes with a single line of code
- Scale effortlessly to millions of time series with multi-GPU and CPU inference
- SOC 2 ready, meeting the security and compliance needs of enterprises

This combination of accuracy, flexibility, and operational readiness makes TimeGPT-2 a powerful foundation for forecasting in retail, logistics, finance, energy, IoT, and beyond.

## Get Early Access

We're currently launching pilot programs with selected organizations. If you'd like to be among the first to leverage TimeGPT-2 in production, [join our waitlist](https://dashboard.nixtla.io/waitlist) to request early access.
