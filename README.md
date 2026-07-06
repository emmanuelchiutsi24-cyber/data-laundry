# Data Laundry 🧺

An automated, flexible data cleaning and profiling workbench built to handle dirty datasets. It generates a detailed analysis profile of your data, executes targeted cleaning pipelines (handling duplicates, outliers, and missing blocks), and lets you download the results with zero structural distortion.

## 🚀 Key Features

*   **Dataset Evaluation:** Generates a summary ledger detailing data types, unique value counts, missing percentages, and statistical outlier spikes.
*   **Selective Deduplication:** Purges exact mirror rows while keeping track of changes.
*   **Winsorization Outlier Treatment:** Caps extreme values at the 1st and 99th percentiles using IQR checking on numerical distributions.
*   **Smart Imputation:** Handles missing data dynamically via mean, median, mode, or KNN structural synthesis based on missingness variance—fully preserving original column order and data types.
*   **Memory-Buffered Export:** Streams clean files back to the browser seamlessly using FastAPI memory blocks.

## 🛠️ Tech Stack

*   **Backend Framework:** FastAPI (Python 2026 stack)
*   **Data Processing Engine:** Polars (High-performance multi-threaded dataframes)
*   **Machine Learning/Stats:** Scikit-learn (`KNNImputer`), NumPy

## 📦 Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/https://github.com/emmanuelchiutsi001-cpu/data-laundry)
