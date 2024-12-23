<<<<<<< HEAD
# Trust_pilot


How to reproduce:

-git clone  

-create conda env  


run in terminal:

conda create --name myenv python=3.12  

conda activate myenv  


pip install -r requirements.txt





run all 3 commands in terminal separetly:  

python3 beau_categories.py  
python3 beau_comp.py  
python3 beau_reviews.py  

after this, 3 .csv files will be returned. the last one is the df which will be mainly used.
=======
# <span style="color:#69b3a2"> Data Scientest Project </span>
Join us in our NLP sentiment analysis project.



By:

- Kjell Hempel
- Felix Wacker
- Fabian Barulli


## <b style="color:#69b3a2"> Table of Contents : </b>

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
  - To gather data:
    
    -`chmod +x run_scraper.sh`

    -`./run_scraper.sh`

 

  

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

>>>>>>> 7a0c1a37d02a0fe8f0690ef407e20e00719fb4f8
