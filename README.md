# Employee-Attrition-Prediction-Project
## Project Overview
Employee attrition is a costly challenge for organizations, leading to higher recruitment expenses, loss of expertise, and reduced productivity. In this project, I developed a predictive model using HR analytics data to identify employees most at risk of leaving. The goal was to provide HR managers with actionable insights that support proactive retention strategies.
### Key Findings
- Attrition Drivers:
- Low satisfaction levels strongly increase the likelihood of leaving.
- Extreme workloads (too few or too many projects, very high hours) are linked to higher turnover.
- Longer tenure without promotion correlates with attrition.
- Low salary groups show higher exit rates.
- Model Performance:
- Baseline (simple logistic regression): ~77% accuracy, but weak at catching leavers.
- Advanced models (XGBoost, Random Forest): up to 95–96% accuracy with strong recall, meaning they successfully identify most employees at risk.
- Final production model: Random Forest with PCA features, chosen for its balance of accuracy and interpretability.
## Business Impact
With this predictive model, HR managers can:
- Identify high risk employees early and intervene with retention programs.
- Optimize workload distribution to reduce burnout and disengagement.
- Adjust compensation and promotion policies to improve retention in vulnerable groups.
- Support workforce planning with data driven insights.
## Tools & Approach
I used Python with libraries such as pandas, seaborn, scikit learn, XGBoost, and Streamlit. The workflow included data cleaning, exploratory analysis, feature engineering, model training, and deployment. The final model was deployed as a Streamlit app with secure login and batch prediction capabilities.
## Conclusion
This project demonstrates that employee attrition can be predicted with high accuracy using HR data. By focusing on workload, satisfaction, tenure, and compensation, organizations can take proactive steps to retain talent, reduce costs, and strengthen employee engagement.

## Login Information
This app uses encrypted cookies to manage user sessions securely.
```
USERNAME = admin 
PASSWORD = admin
```
