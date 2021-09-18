# Credit Card Defaulting Modeling - Yu Miao 
## 1. Statement of Problem: 
Banks were losing money because customers defualted their credit card payments. 

## 2. Client: 
Banks

## 3. Key Business Question: 
How to identify which customers are going to default next month based on the characteristics for that customers? 

## 4. Data source(s): 
https://www.kaggle.com/sakshigoyal7/credit-card-customers

## 5. Business impact of work: 
Approximately 22% of credit card customers are defaulting their payments and median of the bill amounts is 107,912 TWD.
If a bank have 1m credit card customers, about 220 thousands customers are defaulting their payments ≈23.7m TWD 
If 1% of them didn’t pay back → lose 237 thousands TWD every month → 2.8m per year
Save customers from defaulting on payments = Save Money! 
e.g. Save 1% Default Rate → Save 237 thousands TWD every month → 2.8m per year

## 6. Metric: 
Monitoring the trend for customer defaulting rate to see if the trend is decreasing after the credit limit improvement has been launched
Record those who actually defaults and compare the true defaulting name list with the predicted name list

## 7. Methodology:
To begin with, I did feature engineering by transforming AGE and LIMIT BALANCE from numerical variables to categorical variables.
I also transformed age_group, limit_balance_group, sex, education, marriage, etc. from categorical variables into dummy variables (one hot encoding) in order for future modelling. 
Then I split the whole data into train, valid and test data set. By checking the distributions for response variable y (whether the individual default or not on the next payment),
I noticed that the observations for non-defaulters are much more than the observations for defaulters. 
![ditributions-for-sample](https://github.com/UCLA-Stats-404-W21/MIAO-YU/blob/feature/images/distributions%20for%20default%20and%20non-default.png)
To avoid the impact from unbalanced groups, I down sampled the non-defaulters group.
After that, I built up the baseline model by using random forest.
The in sample performance for the random forest model looks good, with accuracy equals to 0.99. 
The out sample performances for the random forest model also looks fair with accuarcy=0.74. 
The f1-score for non-defaulting group is 0.82, which is good. However, the f1-score for defaulting groups is 0.52, which may be due to a relatively high rate of false positives and false negatives.

![table-forrf](https://github.com/UCLA-Stats-404-W21/MIAO-YU/blob/feature/images/performances%20for%20rf%20on%20out%20sample.png)
In order to further improve the performances of our model, the alternative model - K Nearest Neighbor is fitted. 
Unlike random forest, it is hard to get the feature importance for KNN.

The f1 scores on out sample data are similar for both KNN and random forest. Hence, I decides to use the random forest model for modelling.
![table-forknn](https://github.com/UCLA-Stats-404-W21/MIAO-YU/blob/feature/images/performances%20for%20knn%20out%20sample.png)

## 8. Key Findings 
![feature-importance](https://github.com/UCLA-Stats-404-W21/MIAO-YU/blob/feature/images/top%2010%20feature%20importances.png)
From the graph of Top 10 feature importances, the bill amount, pay amount, and balance amount for credit cards in months prior to the next payment
tend to have the greater importances among all features, which make senses in real world as the higher bill amount and balance amount may tend 
to encourage people to default their next credit card payment, while the pay amount in piror months indicate the paying and credit history for those customers.
Surprisingly, the demographic characteristics, such as ages, mariage_status, education levels do not have a large feature importance in the random forest model. 

## 9. How business will use (predicted) model to make decision(s): 
1. Early Warning System to warn account managers that which customers may default in the future (take preventive steps)
2. Automatically lower the current credit limit for customers with high predicted probability to default on next month 

## 10. Potential Further Steps
Conduct additional further research in how demographic factors impact the probability of default payment
 	- Adjust the credit limit for certain groups with high potential to default
	- Used as part of the factors when determining credit limits for customers

## 11. Architecture Diagram
![arc-diagram](https://github.com/UCLA-Stats-404-W21/MIAO-YU/blob/feature/images/arc-diagram.png)
The scoring is batch and the components that shared between my training and scoring is the data processing and feature engineering components.


## 12. Input + output spec
Input: 
- LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit
- SEX: 1/2 (1=male, 2=female)
- EDUCATION : 0/1/2/3/4/5/6 (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
- MARRIAGE: marital status (1=married, 2=single, 3=others)
- AGE: Age in years
- ID: ID of each client
- PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above)
- PAY_2: Repayment status in August, 2005 (scale same as above)
- PAY_3: Repayment status in July, 2005 (scale same as above)
- PAY_4: Repayment status in June, 2005 (scale same as above)
- PAY_5: Repayment status in May, 2005 (scale same as above)
- PAY_6: Repayment status in April, 2005 (scale same as above)
- BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
- BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
- BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
- BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
- BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
- BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
- PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
- PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
- PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
- PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
- PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
- PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)

- Ex:
{'ID':[3.0,4.0],
                     'LIMIT_BAL': [87700.0,90000.0],
                     'SEX': [2.0,2.0],
                     'EDUCATION':[2.0,3.0],
                     'MARRIAGE':[2.0,1.0],
                     'AGE': [34.0,26.0],
                     'PAY_0': [0.0,2.0],
                     'PAY_2': [0.0,2.0],
                     'PAY_3': [1.0,0.0],
                     'PAY_4': [3.0,0.0],
                     'PAY_5': [3.0,0.0],
                     'PAY_6': [2.0,0.0],
                     'BILL_AMT1': [0.0,29239.0],
                     'BILL_AMT2': [3654.0,14027.0],
                     'BILL_AMT3': [1234.0,15549.0],
                     'BILL_AMT4': [1234.0,15549.0],
                     'BILL_AMT5': [1234.0,15549.0],
                     'BILL_AMT6': [1234.0,15549.0],
                     'PAY_AMT1': [1500.0,1500.0],
                     'PAY_AMT2': [1500.0,1500.0],
                     'PAY_AMT3': [3651.0,1000.00],
                     'PAY_AMT4': [1234.0,1000.0],
                     'PAY_AMT5': [2680.0,1000.0],
                     'PAY_AMT6': [2600.0,5000.0]}

Output:
- default.payment.next.month: Default payment (1=yes, 0=no)

- Ex: {0.0,1.0}


## 13. Instructions on how to use the code
1.  Operating System (OS):Windows 10 Pro
2. Anaconda 1.10.0
3. Python 3.70 
4. Steps to follow:
- Clone this repository from github
- start a new virtual environment and install the packages listed in the requirements.txt 
- Input spec (in dictionary format) is defined in main.py and can be edited in main.py (Please follow the types and restrictions for each variables on the above when editing)
- After edited the input spec in main.py, saved it and run main.py to get the predicton based on the input spec you just edited 
- After running the main.py, you will get the result similar to the result listed above in Section 11 



