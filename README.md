# ExtraaLearn Potential Customer Prediction

A machine learning project to predict which leads are more likely to convert to paid customers for ExtraaLearn, an EdTech startup offering programs on cutting-edge technologies.

## Project Overview

The EdTech industry has been surging in the past decade, with the Online Education market forecasted to be worth $286.62bn by 2023 with a compound annual growth rate (CAGR) of 10.26%. With rapid growth and many new companies emerging, identifying high-quality leads that are likely to convert is crucial for efficient resource allocation.

**Problem Statement:** ExtraaLearn generates a large number of leads regularly but needs to identify which leads are more likely to convert to paid customers so they can allocate sales and marketing resources accordingly.

## Objectives

As a data scientist at ExtraaLearn, this project aims to:

1. **Analyze and build ML models** to help identify which leads are more likely to convert to paid customers
2. **Identify key factors** driving the lead conversion process
3. **Create a profile** of leads that are likely to convert
4. **Provide actionable recommendations** for resource allocation and marketing optimization

## Project Structure

```
.
├── venv/                    # Virtual environment (gitignored)
├── requirements.txt         # Python dependencies
├── setup.sh                 # Setup script
├── README.md                # This file
├── data/                    # Data files
│   └── ExtraaLearn.csv     # Leads dataset
├── notebooks/              # Jupyter notebooks
│   └── cuevas25_full_code_potential_customer_prediction.ipynb
└── src/                    # Source code utilities
    └── utils.py            # Helper functions for EDA and model evaluation
```

## Data Description

The dataset contains information about leads and their interactions with ExtraaLearn. The dataset includes **4,612 leads** with the following attributes:

### Data Dictionary

| Column | Description | Values |
|--------|-------------|--------|
| **ID** | Unique identifier for the lead | Numeric |
| **age** | Age of the lead | Numeric |
| **current_occupation** | Current occupation of the lead | 'Professional', 'Unemployed', 'Student' |
| **first_interaction** | How the lead first interacted with ExtraaLearn | 'Website', 'Mobile App' |
| **profile_completed** | Profile completion percentage | 'Low' (0-50%), 'Medium' (50-75%), 'High' (75-100%) |
| **website_visits** | Number of times the lead visited the website | Numeric |
| **time_spent_on_website** | Total time spent on the website (in minutes) | Numeric |
| **page_views_per_visit** | Average number of pages viewed per visit | Numeric |
| **last_activity** | Last interaction between lead and ExtraaLearn | 'Email Activity', 'Phone Activity', 'Website Activity' |
| **print_media_type1** | Flag for newspaper ad exposure | 'Yes', 'No' |
| **print_media_type2** | Flag for magazine ad exposure | 'Yes', 'No' |
| **digital_media** | Flag for digital platform ad exposure | 'Yes', 'No' |
| **educational_channels** | Flag for educational channel exposure | 'Yes', 'No' |
| **referral** | Flag for referral source | 'Yes', 'No' |
| **status** | Target variable - conversion status | 1 (Converted), 0 (Not Converted) |

## Key Insights

### Critical Conversion Factors

1. **Profile Completion (5.6x Impact)**
   - High completion: **41.8%** conversion rate
   - Medium: **18.9%** conversion rate
   - Low: **7.5%** conversion rate

2. **First Interaction Channel (4.4x Difference)**
   - Website: **45.6%** conversion rate
   - Mobile App: **10.5%** conversion rate

3. **Current Occupation**
   - Professionals: **35.5%** conversion (57% of leads) - Highest priority
   - Unemployed: **26.6%** conversion
   - Students: **11.7%** conversion

4. **Last Activity Type**
   - Website Activity: **38.5%** conversion
   - Email Activity: **30.3%** conversion
   - Phone Activity: **21.3%** conversion

5. **Lead Source Channels**
   - Referrals: **67.7%** conversion (highest, but low volume: 93 leads)
   - Print Media (both types): **~32%** conversion
   - Digital Media: **31.9%** conversion
   - Educational Channels: **27.9%** conversion (high volume: 705 leads)

## Models Implemented

The project implements and compares two classification models:

1. **Decision Tree Classifier**
   - Baseline and hyperparameter-tuned versions
   - Hyperparameter tuning using GridSearchCV with 5-fold cross-validation
   - Parameters tuned: `max_depth`, `min_samples_split`, `criterion`

2. **Random Forest Classifier**
   - Baseline and hyperparameter-tuned versions
   - Hyperparameter tuning using GridSearchCV
   - Parameters tuned: `n_estimators`, `max_depth`, `min_samples_split`

### Model Evaluation Metrics

- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of predicted conversions that actually converted
- **Recall**: Proportion of actual conversions correctly identified
- **F1-Score**: Harmonic mean of precision and recall

## Setup Instructions

### Quick Setup (Mac/Linux)

1. Run the setup script:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

### Manual Setup

1. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Mac/Linux
   # OR
   venv\Scripts\activate     # Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Register Jupyter kernel:**
   ```bash
   python -m ipykernel install --user --name=extraalearn_venv --display-name="Python (extraalearn)"
   ```

## Usage

1. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Open the notebook:**
   - Navigate to `notebooks/` directory
   - Open `cuevas25_full_code_potential_customer_prediction.ipynb`

3. **Select the kernel:**
   - Click on the kernel name in the top-right corner
   - Select "Python (extraalearn)" or "Python (hospital_los)" (if using the setup script)

4. **Restart the kernel:**
   - Use `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
   - Type "Jupyter: Restart Kernel"

5. **Update data path:**
   - The notebook references: `../data/ExtraaLearn.csv`

## Dependencies

- **pandas** >= 1.5.0 - Data manipulation and analysis
- **numpy** >= 1.24.0, < 2.0.0 - Numerical computing
- **matplotlib** >= 3.6.0 - Plotting and visualization
- **seaborn** >= 0.12.0 - Statistical visualizations
- **scikit-learn** >= 1.2.0 - Machine learning models and evaluation
- **ipykernel** >= 6.0.0 - Jupyter kernel support
- **jupyter** >= 1.0.0 - Jupyter notebook support

## Business Recommendations

### Immediate Actions

1. **Prioritize Professional Leads**: Allocate 60-70% of sales resources to professional leads (3x higher conversion than students)
2. **Incentivize Profile Completion**: Implement progressive profiling with immediate value offers after 50% completion
3. **Fix Mobile App Experience**: The 10.5% conversion rate is unsustainable - either improve UX or redirect to web
4. **Launch Referral Program**: 67.7% conversion rate justifies significant incentive investment
5. **Optimize Website Engagement**: Add interactive content to increase time-on-site

### Expected Impact

- **15-20% improvement** in overall conversion rate through better lead prioritization
- **25-30% reduction** in cost-per-acquisition by reallocating spend
- **40-50% improvement** in sales team efficiency through automated lead scoring

## Utility Functions

The `src/utils.py` file contains reusable functions for:

- **Univariate Analysis**: `histogram_boxplot()` - Combined boxplot and histogram visualization
- **Bivariate Analysis**: `stacked_barplot()` - Stacked bar charts for categorical analysis
- **Model Evaluation**: 
  - `model_performance_classification()` - Classification metrics
  - `compare_models_classification()` - Side-by-side model comparison
  - `model_performance_regression()` - Regression metrics (for future use)
  - `adj_r2_score()` - Adjusted R-squared calculation
  - `mape_score()` - Mean Absolute Percentage Error

## Notes

- NumPy version is constrained to < 2.0.0 due to compatibility with scipy 1.13.1
- Always restart the kernel after selecting it to ensure correct package versions are loaded
- Data file should be located in the `data/` directory
- The notebook includes comprehensive EDA, feature engineering, and model evaluation sections

## Troubleshooting

### Kernel not showing up
- Ensure `ipykernel` is installed: `pip list | grep ipykernel`
- Re-register the kernel if needed

### Wrong packages being used
- Restart the kernel after selecting it
- Verify with: `import sys; print(sys.executable)`

### File not found errors
- Check that data file is in `data/` directory
- Update notebook path to `../data/ExtraaLearn.csv`

## Author

**cuevas25** - MIT Elective Project

## License

This project is part of an academic elective course at MIT.
