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

# <b style="color:#69b3a2"> Table of Contents : </b>

- Introduction
  -
  - Description of the **project**
  - Description of the goals
  - Table of Contents

&nbsp;

- <span style="font-size: 24px;"> Env Setup</span>

  - After cloning gitrepo
  - Change permissions:
    - `chmod +x setup_env.sh`
  - Run script:
    - `./setup_env.sh`
  
  - Confirm correct installation.

&nbsp;

- Data collection
  -
  - For a more comprehensive explanation on data collection, please see link below:
    - [Totally normal link](https://github.com/fbarulli/trust_me/blob/main/fabian/what_i_learned_scraping.md)

  

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

