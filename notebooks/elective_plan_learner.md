### Phase 1: Importing Necessary Libraries and Data

- **Coding:**
    
    - Import standard libraries (`pandas`, `numpy`, `seaborn`, `matplotlib`).
        
    - Import classification metrics:
    
        - `accuracy_score` (Baseline performance metric)
        
        - `confusion_matrix` (To visualize False Negatives vs. False Positives)
        
        - `precision_score` (Efficiency metric - when we predict conversion, are we right?)
        
        - `recall_score` (Primary "Don't Miss Customers" metric - did we catch all potential conversions?)
        
        - `f1_score` (The balanced metric for model evaluation)
        
        - `classification_report` (Comprehensive metric summary)
        
    - Import classification models:
    
        - `DecisionTreeClassifier`
        
        - `RandomForestClassifier`
        
    - Import model selection and evaluation tools:
    
        - `train_test_split` from `sklearn.model_selection`
        
        - `GridSearchCV` from `sklearn.model_selection` (for model improvement)
        
        - `cross_val_score` (optional, for additional validation)
        
    - Load the leads dataset: `data = pd.read_csv("path/to/leads_data.csv")`
        
    - Copy data to preserve original: `same_data = data.copy()`
        
- **Context & Objective (Markdown):**
    
    - Use the context and objective provided in the Learner Notebook (already matches the project description).
        
- **Data Dictionary (Markdown):**
    
    - Document all features from the leads dataset as provided in the Learner Notebook.
    
    - Clearly indicate `status` as the target variable (0 = unpaid, 1 = converted/paid).

### Phase 2: Data Overview

- **Coding:**
    
    - `data.head()` - View first 5 rows
    
    - `data.shape` - Report dataset dimensions
    
    - `data.info()` - Check data types and null values
    
    - `data.describe().T` - Descriptive statistics for numeric columns
    
    - Check for duplicates: `data.duplicated().sum()`
    
    - Drop identifier column: `data = data.drop(columns=["ID"])` (if ID exists)
    
    - Categorical value counts: Loop through categorical columns and print `value_counts(1)` for each
    
- **Insight Annotation (Data-Driven):**
    
    - **Observations:**
        
        - After `.info()`: Document actual data types, null counts, and memory usage. Note which columns are numeric vs categorical.
        
        - After `.describe()`: Report actual statistics (mean, median, min, max) for numeric features. Identify any obvious outliers or unusual distributions.
        
        - After categorical value counts: Report actual class distribution of `status` variable. Calculate and document the exact ratio (e.g., "The target variable shows a X%/Y% split between converted (1) and unconverted (0) leads"). Based on this actual distribution, determine which metrics are most appropriate (if imbalanced, emphasize Recall; if balanced, accuracy may be sufficient).
        
    - **Sanity Checks:**
        
        - After checking duplicates: Report actual count of duplicate rows found. If duplicates exist, document decision on whether to remove them.
        
        - Document any missing values found and their implications.
        
        - Check for data quality issues (unexpected values, data type mismatches, etc.)

### Phase 3: Exploratory Data Analysis (EDA)

The Hospital notebook uses `histogram_boxplot` and `stacked_barplot`. We will reuse these exact functions.

- **General EDA Approach:**
    
    - Perform thorough analysis of the data in addition to answering the specific questions below.
    
    - Use visualizations to support all findings.
    
- **Univariate Analysis (One variable at a time):**
    
    - **Coding:**
        
        - Define `histogram_boxplot` function (same as Hospital template).
        
        - Plot numeric features: `age`, `website_visits`, `time_spent_on_website`, `page_views_per_visit` using `histogram_boxplot`.
        
    - **Insight Annotation (Data-Driven):**
        
        - For each numeric feature plotted, document actual findings:
        
            - Report exact outlier values found (e.g., "We observe X outliers above Y seconds in time_spent_on_website").
            
            - Note distribution shape (normal, skewed left/right, bimodal, etc.) based on actual histogram.
            
            - Report mean, median, and any notable patterns.
            
- **Bivariate Analysis (Variable vs. Target):**
    
    - **Coding:**
        
        - Create correlation heatmap for numeric variables (including target if encoded as numeric).
        
        - Use `stacked_barplot` function to compare categorical variables against `status`.
        
        - Create barplots showing conversion rates by category (e.g., `sns.barplot` showing mean conversion rate by category).
        
    - **Insight Annotation (Data-Driven):**
        
        - After heatmap: Report actual correlation values. Note which numeric features show strongest correlation with conversion (if any).
        
        - After each stacked barplot: Calculate and report actual conversion rates for each category with exact percentages.
        
- **Answering the 5 Specific Questions (Data-Driven):**
    
    - **Question 1: How does current_occupation affect lead status?**
        
        - **Coding:**
            
            - Use `stacked_barplot(data, "current_occupation", "status")`
            
            - Calculate conversion rates for each occupation category
            
            - Create barplot showing conversion rates by occupation
        
        - **Insight Annotation (Data-Driven):**
            
            - Report actual conversion rates: "Analysis reveals that [Occupation A] converts at [X]%, [Occupation B] converts at [Y]%, and [Occupation C] converts at [Z]%."
            
            - Identify which occupation has highest/lowest conversion with exact percentages.
            
            - Provide interpretation: "This [X-Y]% difference suggests [specific interpretation based on actual numbers and business context]."
            
    - **Question 2: Do the first channels of interaction have an impact on lead status?**
        
        - **Coding:**
            
            - Use `stacked_barplot(data, "first_interaction", "status")`
            
            - Calculate conversion rates for Website vs Mobile App
        
        - **Insight Annotation (Data-Driven):**
            
            - Report actual conversion rates: "Leads who first interacted via [Channel X] convert at [Y]%, while those via [Channel Z] convert at [W]%."
            
            - Calculate and report the difference: "This represents a [X]% difference in conversion rates."
            
            - Provide business insight: "This suggests [interpretation based on actual data]."
            
    - **Question 3: Which way of interaction works best?**
        
        - **Coding:**
            
            - Use `stacked_barplot(data, "last_activity", "status")`
            
            - Calculate conversion rates for Email Activity, Phone Activity, and Website Activity
        
        - **Insight Annotation (Data-Driven):**
            
            - Report actual conversion rates for each activity type: "Email Activity: [X]%, Phone Activity: [Y]%, Website Activity: [Z]%."
            
            - Identify the most effective interaction method with exact percentage.
            
            - Provide recommendation: "Given that [Activity X] shows [Y]% conversion vs [Z]% average, we recommend [specific action based on actual data]."
            
    - **Question 4: Which marketing channels have the highest lead conversion rate?**
        
        - **Coding:**
            
            - Calculate conversion rates for each media flag:
            
                - `print_media_type1` (Newspaper)
                
                - `print_media_type2` (Magazine)
                
                - `digital_media` (Digital platforms)
                
                - `educational_channels` (Education channels)
                
                - `referral` (Referrals)
            
            - Create visualization comparing conversion rates across channels
        
        - **Insight Annotation (Data-Driven):**
            
            - Report actual conversion rates for each channel: "[Channel X]: [Y]%, [Channel Z]: [W]%, etc."
            
            - Identify which channel has highest conversion with exact percentage.
            
            - Provide recommendation: "Given that [Channel X] shows [Y]% conversion vs [Z]% average, we recommend [specific marketing action based on actual data]."
            
    - **Question 5: Does having more details about a prospect increase the chances of conversion?**
        
        - **Coding:**
            
            - Use `stacked_barplot(data, "profile_completed", "status")`
            
            - Calculate conversion rates for Low, Medium, and High profile completion
        
        - **Insight Annotation (Data-Driven):**
            
            - Report actual conversion rates: "Low profile completion: [X]%, Medium: [Y]%, High: [Z]%."
            
            - Analyze the trend: "Conversion rate [increases/decreases/stays constant] as profile completion increases from Low to High."
            
            - Calculate the difference: "High profile completion shows [X]% higher conversion than Low profile completion."
            
            - Provide business insight: "This indicates [interpretation based on actual data]. We should [specific recommendation]."

### Phase 4: Data Preprocessing

- **Coding:**
    
    - **Missing Value Treatment (if needed):**
        
        - Check for missing values: `data.isnull().sum()`
        
        - If missing values exist, document the approach (imputation, removal, etc.) based on actual data patterns.
        
    - **Feature Engineering (if needed):**
        
        - Create any derived features if they add value (e.g., engagement score, interaction frequency, etc.)
        
        - Document rationale for any new features created.
        
    - **Outlier Detection and Treatment (if needed):**
        
        - Identify outliers using IQR method or visualization
        
        - Document outliers found and decision on treatment (keep, cap, remove) based on actual data analysis.
        
    - **Encoding:**
        
        - Use `pd.get_dummies` (One-Hot Encoding) for all categorical columns:
        
            ```python
            data = pd.get_dummies(
                data,
                columns = data.select_dtypes(include = ["object", "category"]).columns.tolist(),
                drop_first = True,
            )
            ```
        
    - **Preparing Data for Modeling:**
        
        - Separate independent variables and target: `X = data.drop('status', axis=1)` and `y = data['status']`
        
        - Split data: `train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)`
        
        - Report train/test shapes
        
- **Insight Annotation:**
    
    - Document any missing values found and how they were handled.
    
    - Document any feature engineering performed and rationale.
    
    - Document outlier treatment decisions.
    
    - Report how many dummy variables were created.
    
    - Report final feature count after encoding.
    
    - Note train/test split ratio and sample sizes.

### Phase 5: Building a Decision Tree Model

- **Coding:**
    
    - Create `model_performance_classification` function that calculates:
    
        - Accuracy
        
        - Precision
        
        - Recall
        
        - F1-score
        
        - Confusion Matrix (as a visual/table)
        
    - Returns a dataframe with all metrics for easy comparison.
    
    - Build Decision Tree Classifier:
    
        - `dt_classifier = DecisionTreeClassifier(random_state=1)`
        
        - Fit model on training data: `dt_classifier.fit(X_train, y_train)`
        
        - Evaluate on test data using classification metrics function
        
- **Insight Annotation (Data-Driven):**
    
    - Report actual metrics: "Decision Tree achieves Accuracy: X%, Precision: Y%, Recall: Z%, F1: W%"
    
    - Display and interpret confusion matrix: "The model correctly predicts [X] conversions and [Y] non-conversions. It has [Z] false positives and [W] false negatives."
    
    - Interpret model performance: "The model shows [strength/weakness] in [specific metric], which is [important/less critical] for lead conversion prediction."

### Phase 6: Model Performance Evaluation and Improvement (Decision Tree)

- **Coding:**
    
    - **Hyperparameter Tuning:**
        
        - Use `GridSearchCV` with classification scoring metric (`'f1'`, `'recall'`, or `'precision'` depending on business priority)
        
        - Define parameter grid (e.g., `max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion`)
        
        - Fit grid search with cross-validation
        
        - Extract best estimator and hyperparameters
        
        - Evaluate tuned model on test set
        
    - **Model Visualization (Optional but Recommended):**
        
        - Plot decision tree with `max_depth=3` for interpretability
        
        - Export tree text representation
        
- **Insight Annotation (Data-Driven):**
    
    - Report best hyperparameters found: "GridSearchCV selected: [actual parameter values]"
    
    - Compare tuned vs untuned performance with exact metrics: "Tuning improved [Metric] from X% to Y%"
    
    - Report final Decision Tree performance: "After tuning, Decision Tree achieves Accuracy: X%, Precision: Y%, Recall: Z%, F1: W%"
    
    - Interpret tree structure if visualized: "The root node splits on [actual feature], indicating this is the most important initial decision for conversion prediction."

### Phase 7: Building a Random Forest Model

- **Coding:**
    
    - Build Random Forest Classifier:
    
        - `rf_classifier = RandomForestClassifier(n_estimators=100, random_state=1)`
        
        - Fit model on training data: `rf_classifier.fit(X_train, y_train)`
        
        - Evaluate on test data using classification metrics function
        
    - **Feature Importance Visualization:**
        
        - Extract feature importances: `importances = rf_classifier.feature_importances_`
        
        - Create horizontal bar chart showing top features (same as Hospital template Cell 100)
        
- **Insight Annotation (Data-Driven):**
    
    - Report actual metrics: "Random Forest achieves Accuracy: X%, Precision: Y%, Recall: Z%, F1: W%"
    
    - Display and interpret confusion matrix.
    
    - Report top 5-10 most important features with their actual importance scores: "The model identifies [Feature 1] (importance: X), [Feature 2] (importance: Y), and [Feature 3] (importance: Z) as the strongest predictors of conversion."
    
    - Provide business interpretation: "This indicates that [interpretation based on actual feature names and importance values]."

### Phase 8: Model Performance Evaluation and Improvement (Random Forest)

- **Coding:**
    
    - **Hyperparameter Tuning:**
        
        - Use `GridSearchCV` with classification scoring metric
        
        - Define parameter grid (e.g., `n_estimators`, `max_depth`, `min_samples_split`, `max_features`)
        
        - Fit grid search with cross-validation
        
        - Extract best estimator and hyperparameters
        
        - Evaluate tuned model on test set
        
    - **Final Model Comparison:**
        
        - Create comparison table with Decision Tree (tuned) and Random Forest (tuned) metrics
        
        - Display as table for easy comparison
        
- **Insight Annotation (Data-Driven):**
    
    - Report best hyperparameters found: "GridSearchCV selected: [actual parameter values]"
    
    - Compare tuned vs untuned performance: "Tuning improved [Metric] from X% to Y%"
    
    - Compare both models: "Decision Tree (tuned) achieves Accuracy: X%, Precision: Y%, Recall: Z%, F1: W%. Random Forest (tuned) achieves Accuracy: A%, Precision: B%, Recall: C%, F1: D%."
    
    - Make final model selection: "Based on comparison, [Model Name] shows best performance with [specific metrics]. This model is chosen because [specific reason based on actual metrics and business requirements]."
    
    - Update feature importance if model was tuned: Report top features from final model.

### Phase 9: Actionable Insights and Recommendations

We will end the notebook with a Markdown section summarizing findings, exactly like the "Business Insights" section at the end of the Hospital notebook.

- **Structure (All insights must be data-driven from actual analysis):**
    
    1. **Key Conversion Drivers:**
        
        - Based on feature importance analysis from Random Forest, list the top factors that drive conversion with actual importance scores
        
        - Example format: "Analysis reveals that [Feature X] is the strongest predictor (importance: Y). Leads with [characteristic] show [Z]% conversion rate vs [W]% for others."
        
    2. **Profile of a High-Value Lead:**
        
        - Use actual EDA findings and feature importance to define the profile
        
        - Example format: "Analysis of top-converting leads shows they have: [Feature 1] = [value/category from actual data], [Feature 2] = [value/category from actual data], [Feature 3] = [value/category from actual data]. This profile represents [X]% of converted leads."
        
    3. **Marketing Channel Effectiveness:**
        
        - Based on actual media flag analysis (Question 4), report which channels drive highest conversion
        
        - Provide specific recommendations: "Given that [Channel X] shows [Y]% conversion vs [Z]% average, we recommend [specific action]."
        
    4. **Interaction Strategy Recommendations:**
        
        - Based on Question 3 analysis, recommend which interaction methods to prioritize
        
        - Example: "Given that [Activity X] shows [Y]% conversion, we recommend [specific action]."
        
    5. **Profile Completion Strategy:**
        
        - Based on Question 5 analysis, provide recommendations on encouraging profile completion
        
        - Example: "Analysis shows that High profile completion leads to [X]% higher conversion. We should [specific action]."
        
    6. **Model Deployment Strategy:**
        
        - Report final model performance metrics
        
        - Explain how the model can be used: "The model can predict conversion probability for new leads, allowing sales team to prioritize [X] leads, potentially improving conversion rate by [estimated impact based on model performance]."
        
        - Note model limitations and monitoring needs
        
    7. **Resource Allocation Recommendations:**
        
        - Based on all findings, provide specific recommendations on where to allocate resources
        
        - Example: "Given that [finding from actual analysis], allocate [X]% more resources to [specific area]."

