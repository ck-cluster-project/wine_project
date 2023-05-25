# Project Goals:
We would be analyzing, exploring the different attributes affecting red and white wine quality. 
We would be creating a model based on to predict the quality of the wine.
This data would not be used on future and/or for real life prediction.

# Project description:
For this project we would exploring the different factors that affect wine quality.
Some of these factors are: fixed acidity, volatile acidity, citric acid, residual sugar, pH, alcohol levels, etc. Focusing on these factors, it would help us identify and predict wine quality. 
# Project planning:
1. Planning:  During this process we asked ourselves important questions about the project, and the division of task among team members. Data planning will be shown in this readme.
2. Acquisition: Data was acquired from data.world published by food.Raw data would be downloaded and a csv for red and white wine data set has been created which would be use to pull the data during this project.
3. Preparation:The red and wine datas would be combines into one dataframe, clean and prepared for exploration.Nulls were handled accordingly and quality assurance was practiced to ensure the validity of each attribute. A column to identify red and white one was created and encoded for moedeling purposes. Outliers were dropped and handled accordingly, and columns were renamed for better identification.
4. Exploration:
5. Evaluation and Modeling:
5. Delivery:
# Initial hypotheses and/or questions you have of the data, ideas:
1. Does Residual Sugar Affect Wine Quality?

2.Does Chlorides Affect Wine Quality?

3.Does Total Sulfur Dioxide Affect Wine Quality?

4.Does Citric Acid Affect Red Wine Quality?


# Data dictionary:
<img width="755" alt="Data Dictionary" src="https://github.com/Keila-Camarillo/wine_project/blob/main/Data%20Dictionary.png">

Data Set Citation: 
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
Modeling wine preferences by data mining from physicochemical properties.
In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.


# Instructions on how someone else can reproduce your project and findings:
1. For an user to succesfully reproduce this project, they must have red and white wine data set downloaded.
2. the wrangle.py, explore.py and evaluate.py files must be downloaded in the same repository/folder as the final_report to run it successfully. 
3. Once all files are download, user may run the final_report notebook.



# Key findings, Recommendations, and Takeaways:
1. The average amount of alcohol in high quality wine is more than the average amount of alcohol in low quality wine.
2. The average amount of chlorides in low quality wine is greater than the average amount of chlorides in high quality wine.
3. The average amount of total sulfur dioxide in low quality wine is greater than the average amount of total sulfur dioxide in high quality wine.
4. The average amount of residual sugar in low quality wine is greater than the average amount of residual sugar in high quality wine.
5.  We would recommend to use our current model to predict the quality of the wine, prioritzing the features selected above.

Given more time:
6.we would like to explore other features and incorporate other clusters into our model.
7. Explore other hyperparameters and different models to improve accuracy.
8. Explore binning the target variable differently or use it as a continous variable to look at regression models.
9. Collect more data such as: location of winery,type barrels used, and year of when wine grapes where harvested.
