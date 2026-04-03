# CS4248 Project: Factual Market Report Generation with Statistical RAG

## Overview
This project explores the automated generation of factual financial market reports from daily market opening tickers. By combining Large Language Models (LLMs) with a custom Statistical Retrieval-Augmented Generation (RAG) pipeline, we aim to mitigate parametric hallucinations and improve the mathematical accuracy of generated financial text.

## Acknowledgements & Attribution
This project builds heavily upon the baseline data-to-text pipeline developed in the [DataTales](https://github.com/YourLinkToDataTales) repository. 
* The baseline dataset compilation, raw table parsing, and evaluation formatting scripts are adapted from their original work. 
* **Our novel contributions** include the engineering of a dynamic `InstanceFactExtractor` for statistical RAG injection, mathematical enrichment of table features, and zero-shot inference optimization on open-weight models.

## Repository Structure
This repository is divided into two primary workflows:

1. **Generating the Benchmark & RAG:** Constructing the dataset, engineering statistical indicators, and customizing the time span and size of the RAG dataset.
2. **Model Inference:** Running inference on different models (e.g., Llama-2, Qwen2.5) with and without RAG on the NUS SoC Computing cluster.

---

## 1. Dataset Construction Pipeline

### 1.1 Manual Data Collection

Due to copyright policies, some data needs to be collected manually:

#### U.S. Treasury Yields
Download from Wall Street Journal (2018/01/01 - 2023/06/30) to `data/raw/wsj/`:

| Bond Yield | URL | File Name |
|------------|-----|-----------|
| 1-Year | [Link](https://www.wsj.com/market-data/quotes/bond/BX/TMUBMUSD01Y/historical-prices) | us_1_year_bond_yield.csv |
| 2-Year | [Link](https://www.wsj.com/market-data/quotes/bond/BX/TMUBMUSD02Y/historical-prices) | us_2_year_bond_yield.csv |
| 3-Year | [Link](https://www.wsj.com/market-data/quotes/bond/BX/TMUBMUSD03Y/historical-prices) | us_3_year_bond_yield.csv |
| 5-Year | [Link](https://www.wsj.com/market-data/quotes/bond/BX/TMUBMUSD05Y/historical-prices) | us_5_year_bond_yield.csv |
| 7-Year | [Link](https://www.wsj.com/market-data/quotes/bond/BX/TMUBMUSD07Y/historical-prices) | us_7_year_bond_yield.csv |
| 10-Year | [Link](https://www.wsj.com/market-data/quotes/bond/BX/TMUBMUSD10Y/historical-prices) | us_10_year_bond_yield.csv |
| 30-Year | [Link](https://www.wsj.com/market-data/quotes/bond/BX/TMUBMUSD30Y/historical-prices) | us_30_year_bond_yield.csv |

### 1.2 Automated Data Collection and Processing

The pipeline consists of three main scripts that process data sequentially:

#### Step 1: Download Market Data & Extract Facts (`download_data.py`)
```bash
# Setup DataBento API key first
export DATABENTO_API_KEY=your_key_here

# Download market data and extract initial statistical facts
python download_data.py
```

**Objectives:**
- Download market data for individual tickers across specified timespans.
- Process data from multiple sources (Yahoo Finance, CME, etc.).
- Combine all market data into a single processed file.

**Output Structure:**
```text
data/
├── tabular_data/
│   └── raw/
│       └── <market>/
│           └── <ticker>.csv    # Individual ticker data
└── intermediate/
    └── processed_data.csv      # Combined market data with statistical facts
```

#### Step 2: Process Table Data & Compute Indicators (`process_table_data.py`)
> **Note:** Before running, ensure you have selected your desired historical time span inside the script (e.g., `1day`, `1week`, `1month`).

```bash
python process_table_data.py
```

**Objectives:**
- Extract historical data for the specified time span.
- **Novel Addition (Mathematical Enrichment):** Calculate additional stock market indicators for every asset within the timeframe, including daily percentage change, weekly change, intraday range, distance from the 20-day Simple Moving Average (SMA-20), and RSI.
- **Novel Addition (Data Categorization):** Classify each asset into categorical variables (e.g., 'Surge/Plunge', 'High/Low Volatility', 'Above/Below 20-SMA') based on the computed indicator values. This prepares the tabular data for the downstream RAG engine.
- Process and format the enriched data for each market report.
- Organize the final tables by market and data source.

**Output Structure:**
```text
data/
└── tabular_data/
    └── report_table_data/
        └── injected/
            └── <historical_time_span>/
                └── <split>/
                    └── <market-report_data_source>/
                        └── <report_date>.csv
```

#### Step 3: Construct the RAG Dataset (`construct_dataset_RAG.py`)
> **Note:** Ensure the configuration inside this script points to the `injected/` tabular data directory and that the `history_span` variable matches the timeline you selected in Step 2.

```bash
python construct_dataset_RAG.py
```

**Objectives:**
- Combine table data with corresponding market reports.
- **Novel Addition:** Dynamically extracts the top statistical facts (via the `InstanceFactExtractor`) and injects them directly into the formatted prompt as a RAG context block.
- Format the final prompts to be ready for model inference or training.

**Output Structure:**
```text
data/
└── processed_dataset/
    └── injected/
        └── <historical_time_span>/
            └── <split>.json        # Contains enriched tables, RAG facts, prompts, and reports
```

## 2. Model Inference Pipeline

This section contains the necessary scripts to evaluate the generated datasets using open-weight Large Language Models. 

**Key Components:**
* **Fine-Tuning & Inference:** Includes complete scripts for LoRA fine-tuning as well as running generation tasks.
* **Supported Models:** Specifically configured and tested for inference on **Llama-2-7b** and **Qwen2.5-7b**.
* **Jupyter Notebook:** A interactive environment to test out RAG generation