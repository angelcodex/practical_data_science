# Potential Customer Predictiond Model

A machine learning project to predict hospital length of stay for patients using admission data and test results.

## Project Structure

```
.
├── venv/                    # Virtual environment (gitignored)
├── requirements.txt         # Python dependencies
├── setup.sh                 # Setup script
├── README.md                # This file
├── data/                    # Data files
│   └── healthcare_data.csv
├── notebooks/              # Jupyter notebooks
│   └── Reference Notebook -- Hospital_LOS_Prediction_Case_Study.ipynb
└── src/                    # Source code (optional utilities)
    └── utils.py
```

## Setup Instructions

### Quick Setup (Mac/Linux)

1. Run the setup script:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

### Manual Setup

1. Create virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Mac/Linux
   # OR
   venv\Scripts\activate     # Windows
   ```

2. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Register Jupyter kernel:
   ```bash
   python -m ipykernel install --user --name=hospital_los_venv --display-name="Python (hospital_los)"
   ```

## Usage

1. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Open the notebook:**
   - Navigate to `notebooks/` directory
   - Open `Reference Notebook -- Hospital_LOS_Prediction_Case_Study.ipynb`

3. **Select the kernel:**
   - Click on the kernel name in the top-right corner
   - Select "Python (hospital_los)"

4. **Restart the kernel:**
   - Use `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
   - Type "Jupyter: Restart Kernel"

5. **Update data path:**
   - The notebook should reference: `../data/healthcare_data.csv`

## Dependencies

- **pandas** >= 1.5.0 - Data manipulation
- **numpy** >= 1.24.0, < 2.0.0 - Numerical computing
- **matplotlib** >= 3.6.0 - Plotting
- **seaborn** >= 0.12.0 - Statistical visualizations
- **scikit-learn** >= 1.2.0 - Machine learning
- **xgboost** >= 2.0.0 - Gradient boosting
- **ipykernel** >= 6.0.0 - Jupyter kernel support
- **jupyter** >= 1.0.0 - Jupyter notebook support

## Project Context

Hospital management is a vital area that gained attention during the COVID-19 pandemic. Inefficient distribution of resources like beds and ventilators can lead to complications. This project aims to predict the length of stay (LOS) of patients before admission, allowing hospitals to plan treatment, resources, and staff allocation more effectively.

## Objective

As a Data Scientist, analyze the data to:
- Identify factors that affect LOS the most
- Build a machine learning model to predict patient LOS
- Provide insights and recommendations to improve healthcare infrastructure and revenue

## Data Dictionary

The dataset contains information recorded during patient admission:

- **patientid**: Patient ID
- **Age**: Range of age of the patient
- **gender**: Gender of the patient
- **Type of Admission**: Trauma, emergency or urgent
- **Severity of Illness**: Extreme, moderate, or minor
- **health_conditions**: Previous health conditions
- **Visitors with Patient**: Number of visitors
- **Insurance**: Health insurance status
- **Admission_Deposit**: Deposit paid during admission
- **Stay (in days)**: Target variable - length of stay
- **Available Extra Rooms in Hospital**: Rooms available during admission
- **Department**: Treating department
- **Ward_Facility_Code**: Ward facility code
- **doctor_name**: Treating doctor
- **staff_available**: Available staff in the ward

## Notes

- NumPy version is constrained to < 2.0.0 due to compatibility with scipy 1.13.1
- Always restart the kernel after selecting it to ensure correct package versions are loaded
- Data file should be located in the `data/` directory

## Troubleshooting

### Kernel not showing up
- Ensure `ipykernel` is installed: `pip list | grep ipykernel`
- Re-register the kernel if needed

### Wrong packages being used
- Restart the kernel after selecting it
- Verify with: `import sys; print(sys.executable)`

### File not found errors
- Check that data file is in `data/` directory
- Update notebook path to `../data/healthcare_data.csv`


