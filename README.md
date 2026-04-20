# ACC102-Track2
ACC102 Python Data Product Track 2
# Financial Distress Prediction: ML Pipeline with Data Leakage Detection

## 1. Problem & User
- **Problem**: How to predict corporate financial distress (negative net income) using accounting metrics while avoiding data leakage and ensuring model validity?
- **Target User**: ACC102 course markers; financial analysts interested in predictive modeling using WRDS Compustat data.

## 2. Data
- **Source**: WRDS Compustat (Wharton Research Data Services)
- **Access Date**: 2026-04-16
- **Time Period**: Fiscal years 2017–2021
- **Raw Data**: 34,445 observations across 8,684 companies (24 columns)
- **Clean Data**: 24,638 observations (71.5% retention after quality checks)
- **Key Fields**:
  - `at`: Total Assets
  - `ni`: Net Income (target variable definition: NI < 0 indicates distress)
  - `lt`: Total Liabilities  
  - `act`: Current Assets
  - `lct`: Current Liabilities
  - `sale`: Sales/Revenue
  - `fyear`: Fiscal Year (for time-series validation)

## 3. Methods (Main Python Steps)
### Data Pipeline
1. **Data Loading**: Imported WRDS Compustat CSV using `pandas`; verified 34,445 rows with 8,684 unique companies.
2. **Data Cleaning**: 
   - Dropped missing values in essential financial fields (`gvkey`, `fyear`, `at`, `ni`, `lt`, `act`, `lct`)
   - Removed invalid records: negative total assets/sales, negative current assets
   - Calculated retention rate: 71.5% data quality pass rate

3. **Feature Engineering**:
   - Calculated financial ratios: ROA, ROE, Leverage (Debt-to-Asset), Current Ratio, Cash Ratio
   - Created `log_at` (Firm Size) and `sales_growth` (YoY growth)
   - Applied Winsorization (1% and 99% percentiles) to handle extreme values

4. **Critical Data Leakage Detection**:
   - **Identified Issue**: ROA perfectly predicts distress (AUC=1.0) because ROA = NI/AT and distress is defined as NI<0, creating mathematical dependency.
   - **Solution**: Excluded ROA/ROE from final model; retained only NI-independent features (`leverage`, `current_ratio`, `log_at`, `sales_growth`).

5. **Modeling Strategy**:
   - **Time-Series Split**: Training set (2017–2019): 4,757 obs; Test set (2020–2021): 13,520 obs (preventing look-ahead bias)
   - **Standardization**: Applied `StandardScaler` to features
   - **Algorithms**: 
     - Logistic Regression (balanced class weights, max_iter=1000)
     - Random Forest (n_estimators=200, max_depth=10, balanced class weights)

6. **Automated Validation Report** (5-Step Rigorous Checks):
   - Data Quality: Verified 24,638 observations, 4 features, 47.9% distress rate
   - Financial Rationality: Confirmed ratio ranges align with accounting theory
   - Coefficient Direction Validation: Verified leverage increases distress probability while liquidity reduces it
   - Model Performance: Logistic AUC = 0.759, Random Forest AUC = 0.791 (reasonable range 0.60–0.95)
   - Confusion Matrix Analysis: Checked prediction distribution and recall rates
7. **Visualization**:
   - Correlation Matrix of Financial Ratios (seaborn heatmap with RdBu_r colormap)
   - Random Forest Feature Importance (horizontal bar chart)

## 4. Key Findings

- **Data Leakage Critical Finding**: ROA achieves perfect prediction (AUC=1.0) due to mathematical dependency with the target variable (NI<0), rendering it invalid for predictive modeling. Final model uses only NI-independent features.

- **Model Performance**: Random Forest outperforms Logistic Regression (AUC 0.791 vs 0.759), suggesting non-linear relationships between financial ratios and distress likelihood.

- **Feature Importance Ranking**: 
  1. Firm Size (Log of Total Assets) – 40% importance
  2. Sales Growth Rate – 28% importance  
  3. Leverage (Debt-to-Asset) – 19% importance
  4. Current Ratio – 13% importance

- **Economic Interpretability**: All model coefficients align with financial theory – higher leverage increases distress probability, while higher liquidity and larger firm size reduce distress risk.

- **Temporal Validation**: Time-series split (train 2017-2019, test 2020-2021) ensures model robustness across different economic periods, including the COVID-19 pandemic onset.
## 5. How to Run

**Prerequisites**: Python 3.8+, Jupyter Notebook/Lab

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn scipy

# Run the analysis
jupyter notebook notebook.ipynb
Data Setup:
 
Download Compustat data from WRDS (2017-2021) or place  compustat_data.csv  in the root directory
 
Update file path in Step 1 if necessary
Expected Runtime: ~3-5 minutes on standard laptop
6. Product Link / Demo
 
GitHub Repository: [TBD]
 
Demo Video: [【ACC102-哔哩哔哩】 https://b23.tv/eXVCPYg]
7. Limitations & Next Steps
Current Limitations:
 
Data Access: Requires WRDS subscription; raw data not publicly distributable due to licensing
 
Static Features: Uses only accounting metrics; excludes market-based predictors and macroeconomic indicators
 
Binary Classification: Simplifies distress to negative net income (NI<0), potentially missing early warning signals
 
Temporal Scope: 2017-2021 includes COVID-19 period which may represent abnormal economic conditions
Next Steps:
 
Incorporate additional NI-independent profitability indicators (e.g., Gross Margin)
 
Extend to multi-class classification (healthy / at-risk / distressed)
 
Add macroeconomic controls (industry fixed effects, year dummies)
 
Deploy as interactive Streamlit app for real-time prediction
AI Disclosure: This project utilized AI tools (ChatGPT/Claude) for code structure suggestions and README drafting. All analytical decisions, financial logic validation, and data leakage analysis were independently verified by the author. Date: 2026-04-16.
