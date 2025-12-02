### Phase 1: Setup & Data Overview

We will replicate the opening cells of the Hospital notebook but adapt the content for EdTech lead conversion classification.

- **Coding:**
    
    - Import standard libraries (`pandas`, `numpy`, `seaborn`, `matplotlib`).
        
    - Import classification metrics instead of regression metrics:
    
        - `accuracy_score` (Baseline performance metric)
        
        - `confusion_matrix` (To visualize False Negatives vs. False Positives)
        
        - `precision_score` (Efficiency metric - when we predict conversion, are we right?)
        
        - `recall_score` (Primary "Don't Miss Customers" metric - did we catch all potential conversions?)
        
        - `f1_score` (The balanced metric we will use for model tuning/comparison)
        
        - `classification_report` (Comprehensive metric summary)
        
    - Import classification models:
    
        - `DecisionTreeClassifier`
        
        - `BaggingClassifier`
        
        - `RandomForestClassifier`
        
        - `AdaBoostClassifier`
        
        - `GradientBoostingClassifier`
        
        - `XGBClassifier` (from xgboost)
        
    - Load the leads dataset.
        
    - Copy data to preserve original: `same_data = data.copy()`
        
- **Context & Objective (Markdown):**
    
    - Rewrite the context to reflect EdTech industry and ExtraaLearn's challenge.
    
    - State objective: Build ML model to identify which leads are likely to convert, find conversion drivers, and create profile of high-value leads.
        
- **Data Dictionary (Markdown):**
    
    - Document all features from the leads dataset (ID, age, current_occupation, first_interaction, profile_completed, website_visits, time_spent_on_website, page_views_per_visit, last_activity, media flags, status).
    
    - Clearly indicate `status` as the target variable (0 = unpaid, 1 = converted/paid).
        
- **Data Overview:**
    
    - **Coding:**
        
        - `data.head()` - View first 5 rows
        
        - `data.shape` - Report dataset dimensions
        
        - `data.info()` - Check data types and null values
        
        - `data.describe().T` - Descriptive statistics for numeric columns
        
        - Check for duplicates: `data.duplicated().sum()`
        
        - Drop identifier column: `data = data.drop(columns=["ID"])` (if ID exists)
        
        - Categorical value counts: Loop through categorical columns and print `value_counts(1)` for each
        
    - **Insight Annotation (Data-Driven):**
        
        - After `.info()`: Document actual data types, null counts, and memory usage. Note which columns are numeric vs categorical.
        
        - After `.describe()`: Report actual statistics (mean, median, min, max) for numeric features. Identify any obvious outliers or unusual distributions.
        
        - After categorical value counts: Report actual class distribution of `status` variable. Calculate and document the exact ratio (e.g., "The target variable shows a X%/Y% split between converted (1) and unconverted (0) leads"). Based on this actual distribution, determine which metrics are most appropriate (if imbalanced, emphasize Recall; if balanced, accuracy may be sufficient).
        
        - After checking duplicates: Report actual count of duplicate rows found.
        
        - Document any missing values found and their implications.

### Phase 2: Exploratory Data Analysis (EDA)

The Hospital notebook uses `histogram_boxplot` and `stacked_barplot`. We will reuse these exact functions.

- **Univariate Analysis (One variable at a time):**
    
    - **Coding:**
        
        - Define `histogram_boxplot` function (same as template).
        
        - Plot numeric features: `age`, `website_visits`, `time_spent_on_website`, `page_views_per_visit` using `histogram_boxplot`.
        
    - **Insight Annotation (Data-Driven):**
        
        - For each numeric feature plotted, document actual findings:
        
            - Report exact outlier values found (e.g., "We observe X outliers above Y seconds in time_spent_on_website").
            
            - Note distribution shape (normal, skewed left/right, bimodal, etc.) based on actual histogram.
            
            - Report mean, median, and any notable patterns.
            
            - For features relevant to conversion, note: "We will check if [feature] correlates with conversion rates in bivariate analysis."
            
- **Bivariate Analysis (Variable vs. Target):**
    
    - **Coding:**
        
        - Create correlation heatmap for numeric variables (including target if encoded as numeric).
        
        - Use `stacked_barplot` function to compare categorical variables against `status`:
        
            - `current_occupation` vs `status`
            
            - `first_interaction` vs `status`
            
            - `profile_completed` vs `status`
            
            - `last_activity` vs `status`
            
        - Create barplots showing conversion rates by category (e.g., `sns.barplot` showing mean conversion rate by `current_occupation`).
        
        - Analyze media flags: Calculate conversion rates for each media type (print_media_type1, print_media_type2, digital_media, educational_channels, referral).
        
    - **Insight Annotation (Data-Driven):**
        
        - After heatmap: Report actual correlation values. Note which numeric features show strongest correlation with conversion (if any).
        
        - After each stacked barplot: Calculate and report actual conversion rates for each category:
        
            - "Analysis reveals that [Category A] converts at [X]%, while [Category B] converts at [Y]%. This [X-Y]% difference suggests [specific interpretation based on actual numbers]."
            
        - After barplots: Identify which categories have highest/lowest conversion rates with exact percentages.
        
        - For media flags: Report which marketing channels show highest conversion rates. Document actual percentages.
        
        - Create actionable insights: "Given that [Feature X] shows [Y]% conversion vs [Z]% for others, we recommend [specific action based on actual data]."

### Phase 3: Data Preprocessing

- **Coding:**
    
    - **Encoding:** Use `pd.get_dummies` (One-Hot Encoding) for all categorical columns, exactly as the Hospital notebook did:
    
        ```python
        data = pd.get_dummies(
            data,
            columns = data.select_dtypes(include = ["object", "category"]).columns.tolist(),
            drop_first = True,
        )
        ```
        
    - **Splitting:** 
        
        - Separate independent variables and target: `X = data.drop('status', axis=1)` and `y = data['status']`
        
        - Split data: `train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)`
        
        - Report train/test shapes
        
- **Insight Annotation:**
    
    - Document how many dummy variables were created.
    
    - Report final feature count after encoding.
    
    - Note train/test split ratio and sample sizes.

### Phase 4: Model Building (The Technical Pivot)

This is where we adapt the template code to fit the Classification problem. We will build all 6 models as in the template.

- **Model Performance Function:**
    
    - **Coding:** Create `model_performance_classification` function that calculates:
    
        - Accuracy
        
        - Precision
        
        - Recall
        
        - F1-score
        
        - Confusion Matrix (as a visual/table)
        
    - Returns a dataframe with all metrics for easy comparison.
        
- **Decision Tree Classifier:**
    
    - **Coding:**
        
        - Change `DecisionTreeRegressor` to `DecisionTreeClassifier`
        
        - Fit model on training data
        
        - Evaluate using classification metrics function
        
    - **Explanation (Markdown):** Include explanation of Decision Trees for classification (same structure as template, adapted for classification context).
        
    - **Visualization:**
        
        - Plot decision tree with `max_depth=3` for interpretability (same as template Cell 66)
        
        - Export tree text representation
        
    - **Insight Annotation (Data-Driven):**
        
        - Report actual metrics: "Decision Tree achieves Accuracy: X%, Precision: Y%, Recall: Z%, F1: W%"
        
        - Interpret tree structure based on actual splits: "The root node splits on [actual feature], indicating this is the most important initial decision for conversion prediction."
        
- **Bagging Classifier:**
    
    - **Coding:**
        
        - Use `BaggingClassifier` with `DecisionTreeClassifier` as base estimator
        
        - Fit and evaluate using classification metrics
        
    - **Explanation (Markdown):** Include explanation of Bagging for classification (adapted from template).
        
    - **Insight Annotation (Data-Driven):** Report actual performance metrics.
        
- **Random Forest Classifier:**
    
    - **Coding:**
        
        - Use `RandomForestClassifier(n_estimators=100, random_state=1)`
        
        - Fit and evaluate using classification metrics
        
    - **Explanation (Markdown):** Include explanation of Random Forest for classification (adapted from template).
        
    - **Insight Annotation (Data-Driven):** Report actual performance metrics.
        
- **AdaBoost Classifier:**
    
    - **Coding:**
        
        - Use `AdaBoostClassifier(random_state=1)`
        
        - Fit and evaluate using classification metrics
        
    - **Explanation (Markdown):** Include explanation of AdaBoost for classification (adapted from template, including equations).
        
    - **Insight Annotation (Data-Driven):** Report actual performance metrics.
        
- **Gradient Boosting Classifier:**
    
    - **Coding:**
        
        - Use `GradientBoostingClassifier(random_state=1)`
        
        - Fit and evaluate using classification metrics
        
    - **Explanation (Markdown):** Include explanation of Gradient Boosting for classification (adapted from template, including equations).
        
    - **Insight Annotation (Data-Driven):** Report actual performance metrics.
        
- **XGBoost Classifier:**
    
    - **Coding:**
        
        - Use `XGBClassifier(random_state=1)` (from xgboost library)
        
        - Fit and evaluate using classification metrics
        
    - **Explanation (Markdown):** Include explanation of XGBoost for classification (adapted from template).
        
    - **Insight Annotation (Data-Driven):** Report actual performance metrics.
        
- **Models' Performance Comparison:**
    
    - **Coding:**
        
        - Create comparison dataframe with all models' metrics (Accuracy, Precision, Recall, F1)
        
        - Display as transposed table for easy comparison
        
    - **Insight Annotation (Data-Driven):**
        
        - Compare all models based on actual metrics
        
        - Identify best performing model(s) with specific numbers
        
        - Discuss trade-offs: "Model X has highest Recall (Y%) but lower Precision (Z%), while Model W has balanced F1 (V%). For lead conversion, we prioritize [Recall/Precision/F1] because [business reason]."
        
        - Note any overfitting concerns by comparing train vs test performance if available
        
        - Consider model complexity and interpretability
        
        - Make preliminary recommendation: "Based on initial comparison, [Model Name] shows best performance with Accuracy: X%, Precision: Y%, Recall: Z%, F1: W%."

### Phase 5: Model Selection & Tuning

- **Choosing Models for Tuning:**
    
    - **Insight Annotation (Markdown):**
        
        - Discuss criteria for model selection (same structure as template Cell 89):
        
            - Evaluation metrics (prioritize Recall for not missing conversions, or Precision for efficiency, or F1 for balance)
            
            - Overfitting concerns (compare train vs test performance)
            
            - Model complexity
            
            - Interpretability (important for business stakeholders)
            
            - Runtime considerations
            
        - State which model(s) will be tuned based on actual comparison results
        
- **Tuning the Model:**
    
    - **Coding:**
        
        - Use `GridSearchCV` with classification scoring metric (`'f1'`, `'recall'`, or `'precision'` depending on business priority)
        
        - Define parameter grid for selected model(s)
        
        - Fit grid search with cross-validation
        
        - Extract best estimator and hyperparameters
        
        - Evaluate tuned model on test set
        
    - **Insight Annotation (Data-Driven):**
        
        - Report best hyperparameters found: "GridSearchCV selected: [actual parameter values]"
        
        - Compare tuned vs untuned performance with exact metrics
        
        - Note improvement: "Tuning improved F1 from X% to Y%" (or note if no significant improvement)
        
- **Final Model Comparison:**
    
    - **Coding:**
        
        - Create final comparison table including tuned model(s)
        
        - Display all models with their metrics
        
    - **Insight Annotation (Data-Driven):**
        
        - Make final model selection: "After tuning, [Model Name] achieves Accuracy: X%, Precision: Y%, Recall: Z%, F1: W%, outperforming alternatives."
        
        - Justify selection: "This model is chosen for production because [specific reason based on actual metrics and business requirements]."
        
- **Visualizing Feature Importance:**
    
    - **Coding:**
        
        - Extract feature importances from best model (Random Forest or XGBoost)
        
        - Create horizontal bar chart showing top features (same as template Cell 100)
        
    - **Insight Annotation (Data-Driven):**
        
        - Report top 5-10 most important features with their actual importance scores
        
        - Interpret findings: "The model identifies [Feature 1] (importance: X), [Feature 2] (importance: Y), and [Feature 3] (importance: Z) as the strongest predictors of conversion."
        
        - Provide business interpretation: "This indicates that [interpretation based on actual feature names and importance values]. For example, [specific insight about what drives conversion]."

### Phase 6: Business Recommendations (The Report)

We will end the notebook with a Markdown section summarizing findings, exactly like the "Business Insights" section at the end of the Hospital notebook (Cell 102-103).

- **Structure (All insights must be data-driven from actual analysis):**
    
    1. **Key Conversion Drivers:**
        
        - Based on feature importance analysis, list the top factors that drive conversion with actual importance scores or conversion rate differences
        
        - Example format: "Analysis reveals that [Feature X] is the strongest predictor (importance: Y). Leads with [characteristic] show [Z]% conversion rate vs [W]% for others."
        
    2. **Profile of a High-Value Lead:**
        
        - Use actual EDA findings and feature importance to define the profile
        
        - Example format: "Analysis of top-converting leads shows they have: [Feature 1] = [value/category from actual data], [Feature 2] = [value/category from actual data], [Feature 3] = [value/category from actual data]. This profile represents [X]% of converted leads."
        
    3. **Marketing Channel Effectiveness:**
        
        - Based on actual media flag analysis, report which channels drive highest conversion
        
        - Provide specific recommendations: "Given that [Channel X] shows [Y]% conversion vs [Z]% average, we recommend [specific action]."
        
    4. **Actionable Recommendations:**
        
        - Prioritize leads: "Based on model predictions, prioritize calls to leads with [specific characteristics from actual data]."
        
        - Resource allocation: "Given that [finding from actual analysis], allocate [X]% more resources to [specific area]."
        
        - Process improvements: "Analysis shows [finding], suggesting we should [specific action]."
        
    5. **Model Deployment Strategy:**
        
        - Report model performance metrics
        
        - Explain how the model can be used: "The model can predict conversion probability for new leads, allowing sales team to prioritize [X] leads, potentially improving conversion rate by [estimated impact based on model performance]."
        
        - Note model limitations and monitoring needs
