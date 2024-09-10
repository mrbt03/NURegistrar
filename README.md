# NURegistrar Class Enrollment Forecasting Project (Capstone Project)

## Overview

This project aims to improve the accuracy of class enrollment forecasts for the McCormick School of Engineering and Applied Sciences at Northwestern University. Accurate forecasts are essential for efficient classroom allocation, reducing last-minute changes, course cancellations, and unmet student demand.

## Models and Methods

We experimented with several predictive models, including:
- **Linear Regression (without regularization)**
- **Linear Regression (with regularization)**
- **XGBoost**

XGBoost was identified as the best performing model due to its ability to capture nonlinear relationships between predictors.

### Key Techniques
- **Latent Semantic Analysis (LSA)**: Used to extract important concepts from course titles, particularly useful for newly offered courses with no historical data.

### Performance
- **Small classes**: Average error of 3 students
- **Medium classes**: Average error of 8 students
- **Large classes**: Average error of 26 students

## Deliverables

### Interactive Dashboard
- An interactive web dashboard for department heads to generate class enrollment forecasts.
- Inputs can be toggled based on desired class specifications.
- Outputs enrollment forecasts and variations based on class start times.
- Allows saving and downloading forecasts to Excel.
- Includes explanatory pages and FAQs for ease of use.

### Additional Resources
- A detailed report explaining our methodology and findings.
- All programming files for transparency and reproducibility.

## Links
- [Interactive Dashboard](https://nuregistrarforecasting.web.app/dashboard)
- [Jupyter Notebook](https://github.com/mrbt03/NURegistrar/blob/main/XGBoostTest.ipynb)
- [Final Report](https://github.com/mrbt03/NURegistrar/blob/c4be6ab30852b7020d7e9c8bfc154afe6e593736/NU%20Registrar%20Final%20Report.pdf)
  
## Credit
The code provided in this repository reflects my significant personal contributions to the capstone project, accounting for the vast majority of the model development. A fellow group member was responsible for the front-end portion of the work. For any questions or additional details, please reach out to Mark Ruiz at markruiz2025@u.northwestern.edu.

