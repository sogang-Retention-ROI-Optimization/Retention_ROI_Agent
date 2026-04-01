# Retention ROI Dashboard + LLM Analyst

## Install

```bash
pip install -r requirements.txt
```

## Generate simulator CSV data

Run the simulator from the project root to create the raw CSV outputs used by the dashboard.

```bash
python3 -c "from src.simulator.pipeline import run_simulation; run_simulation(export=True)"
```

Generated files will be saved under:

```bash
data/raw/
```

Typical outputs include:

- `customer_summary.csv`
- `cohort_retention.csv`
- `customers.csv`
- `events.csv`

## Rebuild cohort retention CSV only

If you already generated simulator data and only want to rebuild the cohort analysis output:

```bash
python3 -m src.simulator.rebuild_cohort_retention
```

This regenerates:

- `data/raw/cohort_retention.csv`

## Run dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard reads the simulator outputs from `data/raw/`.
For the current dashboard views, the most important files are:

- `data/raw/customer_summary.csv`
- `data/raw/cohort_retention.csv`

## Enable LLM summary / Q&A

Recommended: set your OpenAI API key as an environment variable.

```bash
export OPENAI_API_KEY="your-api-key"
streamlit run dashboard/app.py
```

You can also paste the API key in the dashboard sidebar at runtime.

## Added LLM features

- Per-view AI summary below the charts/tables
- Per-view question input for metric-specific answers
- LangChain + ChatOpenAI based integration
- Graceful fallback when API key or packages are missing
