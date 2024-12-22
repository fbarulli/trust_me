# Trust_pilot


How to reproduce:

-git clone  

-create conda env  


run in terminal:

conda create --name myenv python=3.12  

conda activate myenv  


pip install -r requirements.txt





run all commands in terminal separetly:  

python3 beau_categories.py  
python3 beau_comp.py  
python3 beau_all_stars.py  
python3 parse_reviews.py


# <span style="color:#69b3a2"> Data Science Template </span>

This repository contains a jupyter notebook that serves as a template for my data science projects. Feel free to use it for yours as well.

You can find the notebook in this repository as `template.ipynb`.

## <b style="color:#69b3a2"> Table of Contents : </b>

- Introduction
  -
  - Description of the project
  - Description of the goals
  - Table of Contents

&nbsp;

- Setup
  -
  - Installation of required modules <span style="color:#B3697A">(think about using a requirements.txt file)</span>
  - Importing the necessary modules
  - Setup of various settings that will be used throughout the project. Some examples:
    - Configure the figure_format
    - Set up logging if needed
    - Set seaborn / matplotlib themes
    - Set pandas max columns options

&nbsp;

- Data collection
  -
  - Loading the dataset
  - First exploration
    - Head command to see the columns and data
    - Describe command to see the ranges of numerical data
    - Info command as a first quick null check
  - Data Cleaning
    - Transforming data types
    - Handling null values appropriately
    - Merging tables of data to use in EDA

&nbsp;

- EDA: Exploratory Data Analysis
  -
  - Univariate exploration
  - Multivariate exploration
  - Correlations

&nbsp;

- Statistical Analysis
  -
  - Repeat for every hypothesis:
    - Describe the target populations
    - Describe the null and alternative hypothesis
    - Set the significance level
    - Describe assumptions
    - Describe choice of test
    - Describe the results

&nbsp;

- Machine Learning
  -
  - Define one or more prediction goals (repeat next steps for every goal)
    - Load the input data that you need
    - Data preprocessing
      - Address multicollinearity if strong correlations were found during the EDA
      - Think about using dimensionality reduction
      - Label / one-hot encoding
      - Standard scaling
      - Normalization
      - Train - test splitting
    - Model selection and training
      - Explain what model you'll be using
      - Hyperparameter tuning
      - Model training
    - Model evaluation
      - Evaluate model using the metrics of choice.

&nbsp;

- Summary
  -
  - Provide an overview of the entire project with key takeaways

&nbsp;

- Improvements
  -
  - List the possible improvements that you see

