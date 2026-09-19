# Airline Passenger Satisfaction Analytics

Analyzed airline passenger data using SQL and built a dashboard to uncover key factors driving customer satisfaction.

## Overview

This project analyzes 100,000+ airline passenger records to understand what drives customer satisfaction. It combines a SQL-based analytical notebook with a 25-chart interactive Plotly Dash dashboard, covering demographics, service ratings, delays, and loyalty patterns.

## Highlights

- Analyzed 100K+ passenger records via SQL to identify online boarding as the strongest predictor of satisfaction among service factors
- Built a 25-chart interactive Plotly Dash dashboard with KPIs, correlation heatmaps, and clustering across 5 tabs
- Uncovered a 48-point satisfaction gap between business and personal travelers, guiding priorities
- Designed a reusable SQLite pipeline that transforms raw CSV data into query-ready tables for repeatable analysis

## Key Findings

| Insight | Detail |
|---|---|
| Overall satisfaction | Only 43.3% of passengers report being satisfied |
| Strongest driver | Online boarding rating has the highest correlation with satisfaction (r = 0.50) |
| Travel purpose gap | Business travel satisfaction (58.3%) vs. Personal travel (10.2%) — a 48-point gap |
| Delay impact | Minimal — even on-time flights are only 47.3% satisfied |
| Loyalty | Loyal customers fly ~2x farther on average than disloyal customers (1,296 vs. 715 miles) |

## Project Structure

```
├── airline_satisfaction_analysis.ipynb   # SQL-based analysis (25 business questions)
├── airline_satisfaction.db               # SQLite database (flights table + view)
├── app.py                                # Interactive Plotly Dash dashboard
├── train.csv                             # Raw dataset
└── README.md
```

## Tech Stack

- **Python** — pandas, NumPy, SciPy
- **SQL / SQLite** — data storage and analytical queries
- **Plotly & Dash** — interactive visualizations and web dashboard
- **Jupyter Notebook** — exploratory SQL analysis

## Dashboard Preview

The dashboard is organized into 5 tabs:

1. **Distributions** — age, class, customer type, flight distance, delays
2. **Categorical & Bivariate** — satisfaction by class, service ratings ranked, delay effects
3. **Correlations & Heatmaps** — service correlation matrix, clustered heatmap, satisfaction by age/distance
4. **Service Deep Dives** — radar chart (Business vs. Eco), Likert breakdowns, long-haul legroom
5. **Advanced & Faceted** — entertainment facets, satisfaction probability curves, parallel coordinates, waterfall chart

## Getting Started

```bash
pip install pandas numpy plotly dash dash-bootstrap-components scipy

python app.py
```

Then open `http://127.0.0.1:8050` in your browser.

## Dataset

The dataset contains 103,904 airline passenger records with demographic info, 14 service ratings (1–5 scale), flight distance, delay times, and a satisfaction label (`satisfied` / `neutral or dissatisfied`).
