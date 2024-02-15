# deep-learning-challenge
## Challenge 21: Success prediction of funded projects
<br>

### Background
The Alphabet Soup foundation has been funding thousands of projects over the years with various success. The goal of this project is to build a model, based on a large set of historical data, that could predict the success of the failure of an application with an accuracy of at least 75%.<br>
<br>

### Method - ELT
First, we will extract the data from a CSV file provided by the organization. Then we will load it in a Pandas dataframe and analyze it to identify which transformations could make the data better suited for Machine Learning.<br>
Once done, we will randomly split the data between a training and a testing set. We will train a machine learning model and verify its performance on the testing set. We will then try to improve the model by adjusting its parameters and by preprocessing the data further to quantify any possible improvements.<br>
<br>

### Results
#### Dataset analysis and preparation:<br>
The dataset is extracted from a static webpage https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/charity_data.csv. It consists of 12 columns and 34,299 rows with no missing values.<br>
<br>
(show the output of application_df.info())
The target for our model will be the results in the **IS_SUCCESSUL** columns with 0 and 1 indication failure and success respectively. The 11 other columns could potentially become the features, but closer inspection of the data indicates that:

- the data in the EIN column is just a reference number that cannot have an effect on the outcome. We will then drop this column.
- the data in the NAME column might have an effect on the outcome if there is a name-recognition factor, or if the corresponding organizations have more experience in running such ventures, but the name itself will be a meaningless string for the ML model, and we expect that the intrinsic quality of the applicants will be captured by the other features ('APPLICATION_TYPE', 'AFFILIATION', 'CLASSIFICATION', 'USE_CASE', 'ORGANIZATION'). We will then also drop this column.
- The **APPLICATION_TYPE** column has 17 unique values distributed very unevenly with T3 occurring 27,037 time while T17, T15, T29, T14, and T25 occurring less than 5 times each. We will arbitrary bin together all the application types with less than 200 occurrences. It will leave us with 9 categories from the 17 original ones.
- Similarly, the **CLASSIFICATION** column has 71 unique entries with only 5 having more than 1,000 occurrences and most of the other entries being in the single or dual-digit categories. We will arbitrary bin together all the entries with less than 1,000 occurrences. <br>
- We will initially not try to bin the ASK_AMT as the values are very different and could logically have an impact on the outcome.
- We will encode the remaining categorical variables into a binary representation.
- Lastly, we will scale the features by using StandardScaler() from sklearn to remove the mean and scale to unit variance for each column.
<br>

#### Machine Learning Training and Evaluation
We will start with a Sequential model with three layers:
- Layer 1 with 80 nodes and the Rectified Linear Unit (Relu) activation layer.
- Layer 2 with 30 nodes and the Relu activation layer.
- Layer 3 with 1 node and the Sigmoid activation layer.
<br>
The model is compiled with the following parameters:

- loss="binary_crossentropy"
- optimizer="adam"
- metrics=["accuracy"]
We will keep these parameters for all the following model. We will train this first model over 100 epochs.
<br>
The evaluation of the model on the test data leads to an accuracy of 72.97%.<br>
(show the output)
<br>
Remark: The accuracy of the model on the training data was almost 74%, which is close enough to the accuracy obtained with the test data to eliminate the concern of model overfitting.
The notenook is saved as AlphabetSoupCharity.ipynb and the model as AlphabetSoupCharity.h5


#### Second Model
We will experiment the Keras Tuner by letting the Hyperband class identify the best hyperparameters to optimize the model accuracy. We fixed the constraints to be:
- The activation function can be either Relu or Tanh for the first and the hidden layers.
- The first layer can have between 1 and 30 neurons.
- There can be up to 5 hidden layers with up to 30 neurons each.
- The output layer is fixed with 1 node and the Sigmoid activation layer.
- The maximum number of epochs is set to 20 to save time.
<br>
After running 23m 43s in Colab, the best model found yielded an accuracy of 73.26%, which is an improvement from the first attempt, but still falls short of the 75% goal.

The best model is saved as AlphabetSoupCharity_Optimization.h5 and the code can be found at the end of the AlphabetSoupCharity.ipynb notebook.

#### Third Model
In this new attempt, we will first go back to back to the raw data for a closer inspection. The preprocessing will be done in a new notebook AlphabetSoupCharity_Optimization2.ipynb.<br>
- A review of the counts for the successful and failed applications shows them to be at 18,261 and 16,038 respectively. These counts are close enough to eliminate a suspicion of undersampling for one category versus the other that could have created a bias in the model.
- A  review of unique values of the SPECIAL_CONSIDERATIONS feature shows that 34,272 entries have the value 1 and only 27 entries have the value 0. Such a difference makes improbable that this feature has an impact on the model and the column will be dropped with the EIN and the NAME columns.
- A similar review of unique values of the STATUS feature shows that 34,294 entries have the value 1 and only 5 entries have the value 0. Such a difference makes improbable that this feature has an impact on the model and the column will be also dropped.
- A review of the values in the ASK_AMT column indicates a very large spread in the dollar amounts with the minimum and the median being at $5,000.00, the average at $2,769,199.00, and the maximum at $8,597,806,340.00. Based on the statistics, all entries with a ASK_AMT> $11,885 could be considered as an outlier, but there would then be 8,206 outliers and we feel that the success or failure of these high $$ proposals would individually have a high impact on the Alphabet Soup Foundation. Therefore predicting them as accurately as possible is very important, and they cannot be dropped. Moreover, about 55% of these outliers are successful, which is in line with the 53% of successes observed for the whole population.
- We are nevertheless wondering if such a large range in the ASK_AMT values can be efficiently corrected by the StandardScaler method. We will then transform the ASK_AMT column to its decimal logarithm prior to scaling the dataframe.
- The binning of the columns **APPLICATION_TYPE****** and **CLASSIFICATION** to create a **Other** category remains the same.
- The dummy columns are created for the object features.
<br><br>
We train the model a Sequential 3-layer model similar to the one used before:
- Layer 1 with 80 nodes and the Rectified Linear Unit (Relu) activation layer.
- Layer 2 with 30 nodes and the Relu activation layer.
- Layer 3 with 1 node and the Sigmoid activation layer.
<br>
The evaluation of the model on the test data leads to an accuracy of 72.58%, which is worse than what we obtained before ().<br>
(show the output)
The model is saved as AlphabetSoupCharity_Optimization2.h5
<br>
<br>

#### Fourth Model
We will run the same dataset through a slightly more complex Keras Tuner. We fixed the constraints to be:
- The activation function can be either Relu, Tanh, ot sigmoid for the first and the hidden layers.
- The first layer can have between 1 and 30 neurons.
- There can be up to 5 hidden layers with up to 30 neurons each.
- The output layer is fixed with 1 node and the Sigmoid activation layer.
- The maximum number of epochs is set to 40.
<br>
After running 1h 18m 19s in Colab, the best model found yielded an accuracy of 73.41%, which is the best result so far, but still falls short of the 75% goal.

The best model is saved as AlphabetSoupCharity_Optimization3.h5

#### Fifth Model
Next, we will explore the effect of the scaling method applied to the last dataset. The reasoning is that the StandardScaler returns normalize values centered around zero (i.e. with negative values), while the Relu and Sigmoid activation functions are between 0 and 1. We will use the MinMaxScaler method that, by construction, yields a output between 0 and 1 too. The notebook is saved under AlphabetSoupCharity3.ipynb<br>
We  directly ran the Keras Tuner with the same constrains as for the fourth model, and the best model had an accuracy of  73.42%, which is only marginally better than the fourth model.<br>
The notenook is saved as AlphabetSoupCharity_Optimization3.ipynb and the model as AlphabetSoupCharity_Optimization4.h5

### Summary
While we were not able to achieve an accuracy rate over 75%, our best model after additional preprocessing achieved 73.42% which 0.44 percentage point above the result obtained with the more basic approach used for the first model.










