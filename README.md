# Machine-Learning-Projects
Projects in this Repository

1. Ads-CTR-Optimization
2. Market Basket Optimization
3. NLP Restraunt Review Analysis
4. Poetry Generation using LSTM
5. Churn Modelling using ANN
6. Clustering of Mall Customers
7. CNN Classifier
8. Binary Classifier
9. Startup Company Profit Prediction
10. Employee Salary Prediction


# Ads CTR Optimisation using Reinforcement Learning

The Dataset contains information about ten differnt types of ad with differnt types of designs for the same product or service displayed to differnt users online once they connect to a certain website 
or to a search engine . Each Row corresponds to a differnt user on the internet. Since there are 10000 rows, we have data of ten thousand users.
0-> User clciked on the ad
1-> User din't click on the ad

In reality, what happens is that users connect one by one to the webpage and for each of them, we successively show them the ad. So everything happens in real time.
It's a dynamic real time process. It's not a static process with a static dataset which was recorded over a certain period of time.

This dataset is a simulation in the sense that each time a user connects to the webpage, it tells us even if we wouldn't know in reality on which ad the user would click on and this is the only way we can actually run the UCB algorithm or the Thomson Sampling algorithm 
to figure out the ad that has the highest conversion rate.The users here are represented as rounds so we need to figure out in a minimum number of rounds, which ad converts to the most meaning, which is the best ad to which the users are most attracted to.
And that's why we need a stronger algorithm than a simple statistics algorithm.

### Upper Confidence Bound Algorithm
 
 #### Step 1
 At each round n, we consider two numbers for each ad i:
 
 N<sub>i</sub>(n) - the number of times the ad i was selected up to round n
 
 R<sub>i</sub>(n) - the sum of rewards of the ad i up to round n
    
 #### Step 2 
 From these two numbers we compute:
 
   The average reward of ad i upto round n 
   
   r_avg<sub>i</sub>(n) = R<sub>i</sub>(n) / N<sub>i</sub>(n)
   
   The confidence interval [ r_avg<sub>i</sub>(n) - &Delta;<sub>i</sub>(n) , r_avg<sub>i</sub>(n) + &Delta;<sub>i</sub>(n) ] at 
   round n with 
   
   &Delta;<sub>i</sub>(n) = &radic; ( 3log(n) / 2 N<sub>i</sub>(n) )
         
 #### Step 3
 We select the ad i that has the maximum UCB r_avg<sub>i</sub>(n) + &Delta;<sub>i</sub>(n)

 ### Thompson Sampling Algorithm

#### Step 1
 At each round n, we consider two numbers for each ad i:
 
 N<sub>i</sub><sup>1</sup>(n) - the number of times the ad i got reward 1 up to round n
 
 N<sub>i</sub><sup>0</sup>(n) - the number of times the ad i got reward 1 up to round n
 
 #### Step 2
 For each ad i, we take a random draw from the distribution below:
 
 &theta;<sub>i</sub>(n) = &beta;( N<sub>i</sub><sup>1</sup>(n) + 1 , N<sub>i</sub><sup>0</sup>(n) + 1 )
 
 #### Step 3 
 We select the ad that has the highest &theta;<sub>i</sub>(n)
 
 # Market Basket Optimization using Association Rule Learning

In a store, all vegetables are placed in the same aisle, all dairy items are placed together and cosmetics form another set of such groups. Investing time and resources on deliberate product placements like this not only reduces a customer’s shopping time, but also reminds the customer of what relevant items (s)he might be interested in buying, thus helping stores cross-sell in the process. Association rules help uncover all such relationships between items from huge databases.

Rules do not extract an individual’s preference, rather find relationships between set of elements of every distinct transaction. This is what makes them different from collaborative filtering.Rules do not tie back a users’ different transactions over time to identify relationships. List of items with unique transaction IDs (from all users) are studied as one group. This is helpful in placement of products on aisles, in recommending items on e-commerce websites.

So Identifying the best deals can maximize the sales and profit.

# Restraunt Review Analysis using Natural Language Processing 

Natural Language Processing (or NLP) is applying Machine Learning models to text and language. Teaching machines to understand what is said in spoken and written word is the focus of Natural Language Processing. Whenever you dictate something into your iPhone / Android device that is then converted to text, that’s an NLP algorithm in action.

A very well-known model in NLP Bag of Words model is used for the project. It is a model used to preprocess the texts to classify before fitting the classification algorithms on the observations containing the texts.

# Poetry Generation using LSTM

Used a Bi-Directional Tensorflow LSTM model to generate poetry by training on the wreath corpus.

# Churn Modelling using Artificial Neural Network

It is no secret that customer retention is a top priority for many companies;
acquiring new customers can be several times more expensive than retaining existing ones. Furthermore, gaining an understanding of the reasons customers churn and estimating the risk associated with individual customers are both powerful components of designing a data-driven retention strategy. A churn model can be the tool that brings these elements together and provides insights and outputs that drive decision making across an organization.

churn rate is calculated by dividing the number of customer cancellations within a time period by the number of active customers at the start of that period. Very valuable insights can be gathered from this simple analysis — for example, the overall churn rate can provide a benchmark against which to measure the impact of a model. And knowing how churn rate varies by time of the week or month, product line, or customer cohort can help inform simple customer segments for targeting as well.
However, churn is often needed at more granular customer level. Customers vary in their behaviors and preferences, which in turn influence their satisfaction or desire to cancel service. Therefore, a cohort-based churn rate may not be enough for precise targeting or real-time risk prediction. This is where churn modeling is usually most useful.
The output of a predictive churn model is a measure of the immediate or future risk of a customer cancellation. This is what the term "churn modeling" most often refers to.

So we will build a Artificial Neural Network using Tensorflow to predict whether a customer of the bank will leave the bank or not.

# Clustering using K-means and Hierarchial clustering methods

The data set contains the survey information of customers of the mall. It contains CustomerID, Genre, Age, Annual Income (k$), Spending Score (1-100).

Since this is unsupervised learning we are not predicting anything but instead based on this information we will be clustering them into 5 differnt types of clusters which are then visualized in the notebook.

# Cats vs Dogs classifier using Tensorflow Convnet

The Dataset contains images of dogs and cats. Each image in the dataset has a dimension of 64x64. Using Tensorflow we train the Convolutional neural network on these images to classiy the images into dogs or cats.

 The training set consists of 4000 images of dogs and 4000 images of cats.

 The test set consists of 1000 images of dogs and 1000 images of cats.

The model consists of 2 convolutional layers with maxpool layer inserted in between.We flatten the feature maps generated by the second convolutional layer and pass them through the fully connected layer with 128 units or neurons to get the final probabilities in the output layer.

The model was trained for 25 epochs , with a batch size of 32 images using `adam` optimizer and `binary_crossentropy` as the loss function. The training time took around 5 minutes and the model acheived an accuracy of 81%

Try out the model by giving your own images by adding it to the ```single_prediction``` folder and changing the name of the file in ```Making a single prediction``` section in the code and see what the model predicts.

The Model can be trained to classify any two classes of images since its a Binary Classifier, and experiment with the hyperparameters to get high accuracies.
 
 # Binary Classifier using Naive Bayes and Kernel SVM 

The Data Set contains information on customer's of a car company. It contains their age, salary, and whether or not they bought the SUV car previously. 
So based on these values we are Predicting whether a person will buy the SUV car or not. 

# Startup Company Profit Prediction

The dataset has columns R&D spend, Administration spend, Marketing spend, State, Profit.

Using Multiple Linear Regression we can predict the future profit of a Startup Company based on the R&D spend, Administration spend, Marketing spend, State values.
The model gave an accuracy of 93% so, its not overfitted with the training data set.

# Employee Salary prediction 

### Position and the level of employees in the company
**Business Analyst**     1

**Junior Consultant**    2

**Senior Consultant**    3

**Manager**              4

**Country Manager**      5

**Region Manager**       6

**Partner**              7

**Senior Partner**       8

**C-level**              9

**CEO**                 10

Predicting the salary of employees based on his position in the company using Support Vector Regression (SVR) and Random Forest Regression.

Based on the visualization of the graphs and the predicted results we see that Random forest Regression with number of trees in the forest i.e. `n_estimators = 10` performs better than Support Vector Regression for the particular Data Set.






