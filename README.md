# NURegistrar Class Enrollment Forecasting Project

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
- [Interactive Dashboard](#) (link to be added)
- [Jupyter Notebook](#) (link to be added)

## Contact
For any questions or further information, please contact [Your Name] at [Your Email].
