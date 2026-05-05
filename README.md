# FinRecipe: Extending TimeRecipe for Financial Time Series Forecasting

This repository extends the TimeRecipe modular forecasting framework with financial domain specific components. It introduces new modules into the five stages of the forecasting pipeline to address the unique a statistical properties of financial time series.

Branch Structure:
The main branch contains this README file and the basic TimeRecipe framework. All FinRecipe financial domain extensions live in the add-modules branch.

The integrate-all branch combines add-modules with the financial datasets and test scrips from other branches.

The other branches were used for testing and collaboration throughout the semester. There is testing code within each but it is not relevant for the final product.

New Components: 

The following new components were added to the TimeRecipe framework.

Preprocessing - Log Transform, Volatility Normalization, Fractional Differencing, Fourier Seasonal Demeaning

Embedding - Residual Embedding

Feed Forward Modeling - Gated Residual Network, N-BEATS, Regime Switching MLP, MAGN

Datasets:

Four financial datasets were added to the repo root to serve as financial benchmarks. Each of which cover a different type of financial time series or were gathered from a different region.

    - VIX Futures
    
    - US Investment Grade OAS
    
    - European High Yield OAS
    
    - 30 Year US Mortgage Rates

Installation:

    git clone https://github.com/SachiGoel27/FinancialTimeSeries.git
    
    cd FinancialTimeSeries
    
    git checkout add-modules
    
    cd TimeRecipe
    
    pip install -r requirements.txt

How to Run:
To run all of our outlines experiments, do the following. Note this will run a complete iteration through all of the modules on all four datasets. This will not do one run of the pipline with an inputted specified framework.

cd .. # If inside TimeRecipe/, cd back to the repo root

python run_financial_experiments.py --dry_run # Preview

python run_financial_experiments.py # Full experiment list
