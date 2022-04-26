{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h1>Understanding Train Test Split using Scikit-Learn (Python)</h1> "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![](images/TrainTestProcedure.png)\n",
    "\n",
    "A goal of supervised learning is to build a model that performs well on new data. If you have new data, it’s a good idea to see how your model performs on it. The problem is that you may not have new data, but you can simulate this experience with a procedure like train test split. This tutorial includes:\n",
    "\n",
    "* What is the Train Test Split Procedure\n",
    "* Using Train Test Split to Tune Models using Python\n",
    "* The Bias-variance Tradeoff\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2>What is the Train Test Split Procedure</h2>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![](images/TrainTestProcedure.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "train test split is a model validation procedure that allows you to simulate how a model would perform on new/unseen data. Here is how the procedure works.\n",
    "\n",
    "0. Make sure your data is arranged into a format acceptable for train test split. In scikit-learn, this consists of separating your full dataset into Features and Target. \n",
    "1. Split the dataset into two pieces: a training set and a testing set. This consists of randomly selecting about 75% (you can vary this) of the rows and putting them into your training set and putting the remaining 25% to your test set. Note that the colors in “Features” and “Target” indicate where their data will go (“X_train”, “X_test”, “y_train”, “y_test”) for a particular train test split.\n",
    "2. Train the model on the training set. This is “X_train” and “y_train” in the image. \n",
    "3. Test the model on the testing set (“X_test” and “y_test” in the image) and evaluate the performance. \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2>Consequences of NOT using Train Test Split</h2>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "You could try not using train test split and <b>train and test the model on the same data</b>. I don’t recommend this approach as it doesn’t simulate how a model would perform on new/unseen data and it tends to reward overly complex models that overfit on the dataset. \n",
    "\n",
    "The steps below go over how this inadvisable process works. \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![](images/NotUsingTrainTestSplit.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "0. Make sure your data is arranged into a format acceptable for train test split. In scikit-learn, this consists of separating your full dataset into Features and Target.\n",
    "1. Train the model on “Features” and “Target”. \n",
    "2. Test the model on “Features” and “Target” and evaluate the performance.\n",
    "\n",
    "It is important to again emphasize that training on an entire data set and then testing on that same dataset can lead to overfitting. You might find the image below useful in explaining what overfitting is.  The green squiggly line best follows the training data. The problem is that it is likely overfitting on the training data meaning it is likely to perform worse on unseen/new data. [Image contributed by Chabacano to Wikipedia (CC BY-SA 4.0)](https://en.wikipedia.org/wiki/Overfitting#/media/File:Overfitting.svg)(https://creativecommons.org/licenses/by-sa/4.0/). \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![](images/Overfitting.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2>Using Train Test Split to Tune Models using Python\n",
    "</h2>\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![](images/TrainTestRepeat.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This section is about the practical application of train test split to predicting home prices. It goes all the way from importing a dataset to performing a train test split to hyperparameter tuning (change hyperparameters in the image above is also known as hyperparameter tuning) a decision tree regressor to predict home prices and more. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3>Import Libraries</h3>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![](images/PythonLibraries.jpg)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Python has a lot of libraries that can help you accomplish your data science goals (the image above is likely from [Reddit](https://www.reddit.com/r/ProgrammerHumor/comments/6a59fw/import_essay/)) including scikit-learn, pandas, and NumPy which the code below imports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn import tree\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.tree import DecisionTreeRegressor"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3>Load the Dataset\n",
    "</h3>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Kaggle hosts a dataset which contains the price at which houses were sold for King County, which includes Seattle between May 2014 and May 2015. You can download the dataset from [Kaggle](https://www.kaggle.com/harlfoxem/housesalesprediction) or load it from my [GitHub](https://raw.githubusercontent.com/mGalarnyk/Tutorial_Data/master/King_County/kingCountyHouseData.csv). The code below loads the dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>bedrooms</th>\n",
       "      <th>bathrooms</th>\n",
       "      <th>sqft_living</th>\n",
       "      <th>sqft_lot</th>\n",
       "      <th>floors</th>\n",
       "      <th>price</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>3</td>\n",
       "      <td>1.00</td>\n",
       "      <td>1180</td>\n",
       "      <td>5650</td>\n",
       "      <td>1.0</td>\n",
       "      <td>221900.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>3</td>\n",
       "      <td>2.25</td>\n",
       "      <td>2570</td>\n",
       "      <td>7242</td>\n",
       "      <td>2.0</td>\n",
       "      <td>538000.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>1.00</td>\n",
       "      <td>770</td>\n",
       "      <td>10000</td>\n",
       "      <td>1.0</td>\n",
       "      <td>180000.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>3.00</td>\n",
       "      <td>1960</td>\n",
       "      <td>5000</td>\n",
       "      <td>1.0</td>\n",
       "      <td>604000.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3</td>\n",
       "      <td>2.00</td>\n",
       "      <td>1680</td>\n",
       "      <td>8080</td>\n",
       "      <td>1.0</td>\n",
       "      <td>510000.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>4</td>\n",
       "      <td>4.50</td>\n",
       "      <td>5420</td>\n",
       "      <td>101930</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1225000.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>3</td>\n",
       "      <td>2.25</td>\n",
       "      <td>1715</td>\n",
       "      <td>6819</td>\n",
       "      <td>2.0</td>\n",
       "      <td>257500.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>3</td>\n",
       "      <td>1.50</td>\n",
       "      <td>1060</td>\n",
       "      <td>9711</td>\n",
       "      <td>1.0</td>\n",
       "      <td>291850.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>3</td>\n",
       "      <td>1.00</td>\n",
       "      <td>1780</td>\n",
       "      <td>7470</td>\n",
       "      <td>1.0</td>\n",
       "      <td>229500.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>3</td>\n",
       "      <td>2.50</td>\n",
       "      <td>1890</td>\n",
       "      <td>6560</td>\n",
       "      <td>2.0</td>\n",
       "      <td>323000.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   bedrooms  bathrooms  sqft_living  sqft_lot  floors      price\n",
       "0         3       1.00         1180      5650     1.0   221900.0\n",
       "1         3       2.25         2570      7242     2.0   538000.0\n",
       "2         2       1.00          770     10000     1.0   180000.0\n",
       "3         4       3.00         1960      5000     1.0   604000.0\n",
       "4         3       2.00         1680      8080     1.0   510000.0\n",
       "5         4       4.50         5420    101930     1.0  1225000.0\n",
       "6         3       2.25         1715      6819     2.0   257500.0\n",
       "7         3       1.50         1060      9711     1.0   291850.0\n",
       "8         3       1.00         1780      7470     1.0   229500.0\n",
       "9         3       2.50         1890      6560     2.0   323000.0"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "url = 'https://raw.githubusercontent.com/mGalarnyk/Tutorial_Data/master/King_County/kingCountyHouseData.csv'\n",
    "df = pd.read_csv(url)\n",
    "# Selecting columns I am interested in\n",
    "columns = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','price']\n",
    "df = df.loc[:, columns]\n",
    "df.head(10)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3>Arrange Data into Features and Target</h3>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Scikit-Learn’s train_test_split expects data in the form of features and target. In scikit-learn, a features matrix is a two-dimensional grid of data where rows represent samples and columns represent features. A target is what you want to predict from the data. This tutorial uses ‘price’ as a target. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors']\n",
    "X = df.loc[:, features]\n",
    "y = df.loc[:, ['price']]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![](images/KingCountyArrangeData.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3>Split Data into Training and Testing Sets (train test split)\n",
    "</h3>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![](images/KingCountyTrainTestSplit.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The colors in the image above indicate which variable (X_train, X_test, y_train, y_test) from the original dataframe df will go to for our particular train test split (random_state = 0). "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In the code below, train_test_split splits the data and returns a list which contains four NumPy arrays. train_size = .75 puts 75% of the data into a training set and the remaining 25% into a testing set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size = .75)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The image below shows the number of rows and columns the variables contain using the shape attribute before and after the train test split. 75 percent of the rows went to the training set (16209/ 21613 = .75) and 25 percent went to the test set (5404 / 21613 = .25)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![](images/KingCountyShape.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3>Understanding random_state</h3>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![](images/KingCountyRandomState.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The random_state is a pseudo-random number parameter that allows you to reproduce the same exact train test split each time you run the code. The image above shows that if you select a different value for random state, different information would go to X_train, X_test, y_train, and y_test. There are a number of reasons why people use random_state including software testing, tutorials (like this one), and talks. However, it is recommended you remove it if you are trying to see how well a model generalizes to new data."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3>Creating and Training a Model with Scikit-learn</h3>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Step 1:</b> Import the model you want to use.\n",
    "\n",
    "In scikit-learn, all machine learning models are implemented as Python classes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.tree import DecisionTreeRegressor"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Step 2:</b> Make an instance of the model\n",
    "\n",
    "In the code below, I set the hyperparameter max_depth = 2 to preprune my tree to make sure it doesn’t have a depth greater than 2. I should note the next section of the tutorial will go over how to choose an optimal max_depth for your tree.\n",
    "\n",
    "Also note that in my code below, I made random_state = 0 so that you can get the same results as me."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "reg = DecisionTreeRegressor(max_depth = 2, random_state = 0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Step 3:</b> Train the model on the data, storing the information learned from the data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DecisionTreeRegressor(max_depth=2, random_state=0)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "reg.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Step 4:</b> Predict labels of unseen (test) data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 406622.58288211, 1095030.54807692,  406622.58288211,\n",
       "        406622.58288211,  657115.94280443,  406622.58288211,\n",
       "        406622.58288211,  657115.94280443,  657115.94280443,\n",
       "       1095030.54807692])"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# You can predict for multiple observations\n",
    "reg.predict(X_test[0:10])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For the multiple predictions above, notice how many times some of the predictions are repeated. If you are wondering why, I encourage you to check out the code below which will start by looking at a single observation/house and then proceed to look at how the model makes its prediction."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>bedrooms</th>\n",
       "      <th>bathrooms</th>\n",
       "      <th>sqft_living</th>\n",
       "      <th>sqft_lot</th>\n",
       "      <th>floors</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>17384</th>\n",
       "      <td>2</td>\n",
       "      <td>1.5</td>\n",
       "      <td>1430</td>\n",
       "      <td>1650</td>\n",
       "      <td>3.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       bedrooms  bathrooms  sqft_living  sqft_lot  floors\n",
       "17384         2        1.5         1430      1650     3.0"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_test.head(1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The code below shows how to make a prediction for that single observation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([406622.58288211])"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# You can also predict for 1 observation.\n",
    "reg.predict(X_test.iloc[0].values.reshape(1,-1))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The image below shows how the trained model makes a prediction for the one observation."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![](images/HousePredictions.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "If you are curious how these sorts of diagrams are made, consider checking out my tutorial [Visualizing Decision Trees using Graphviz and Matplotlib](https://towardsdatascience.com/visualizing-decision-trees-with-python-scikit-learn-graphviz-matplotlib-1c50b4aa68dc)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3>Measuring Model Performance</h3>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![](images/CoefficientDetermination.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "While there are other ways of measuring model performance (root-mean-square error, mean absolute error, mean absolute error, etc), we are going to keep this simple and use R² otherwise known as the coefficient of determination as our metric. The best possible score is 1.0. A constant model that would always predict the mean value of price would get a R² score of 0.0 (interestingly it is possible to get a negative R² on the test set). The code below uses the trained model’s score method to return the R² of the model that was evaluated on the test set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.4380405655348807\n"
     ]
    }
   ],
   "source": [
    "score = reg.score(X_test, y_test)\n",
    "print(score)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "You might be wondering if our R² above is good for our model. In general the higher the R², the better the model fits the data. Determining whether a model is performing well can also depend on your field of study. Something harder to predict will in general have a lower R². My argument below is that for housing data, we should have a higher R² based solely on our data.\n",
    "\n",
    "Here is why. Domain experts generally agree that one of the most important factors in housing prices is location. After all, if you are looking for a home, most likely you care where it is located. As you can see in the trained model below, the decision tree only incorporates sqft_living.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![](images/treeNoCustomarrows.png)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize Decision Tree using Graphviz\n",
    "\"\"\"\n",
    "tree.export_graphviz(reg,\n",
    "                     out_file=\"images/temp.dot\",\n",
    "                     feature_names = features,\n",
    "                     filled = True)\n",
    "\"\"\"\n",
    "\n",
    "# You need to have graphviz installed and added to your path for this \n",
    "# to work\n",
    "#!dot -Tpng -Gdpi=300 images/temp.dot -o images/temp.png"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'\\nfig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)\\ntree.plot_tree(reg,\\n              feature_names = features,\\n              filled = True);\\n'"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# code that generates matplotlib based decision trees. \n",
    "\"\"\"\n",
    "fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)\n",
    "tree.plot_tree(reg,\n",
    "              feature_names = features,\n",
    "              filled = True);\n",
    "\"\"\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Even if the model was performing very well, it is unlikely that our model would get buy-in from stakeholders or coworkers as traditionally speaking, there is more to homes than sqft_living.\n",
    "\n",
    "Note that the original dataset has location information like ‘lat’ and ‘long’. The image below visualizes the price percentile of all the houses in the dataset based on ‘lat’ and ‘long’ (‘lat’ ‘long’ wasn’t included in data which the model trained on). There is definitely a relationship between home price and location.\n",
    "\n",
    "A way to improve the model would be to make it incorporate location information (‘lat’, ‘long’) as it is likely places like Zillow found a way to incorporate that into their models."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![](images/KingCountyHousingPrices.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3>Tuning the max_depth of a Tree</h3>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The R² for the model trained earlier in the tutorial was about .438. However, suppose we want to improve the performance so that we can better make predictions on unseen data. While we could definitely add more features like lat long to the model or increase the number of rows in the dataset (find more houses), another way to improve performance is through hyperparameter tuning which involves selecting the optimal values of tuning parameters for a machine learning problem. These tuning parameters are often called hyperparameters. Before doing hyperparameter tuning, we need to take a step back and briefly go over the difference between parameters and hyperparameters. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b>Parameters vs hyperparameters</b>\n",
    "\n",
    "A machine learning algorithm estimates model parameters for a given data set and updates these values as it continues to learn. You can think of a model parameter as a learned value from applying the fitting process. For example, in logistic regression you have model coefficients. In a neural network, you can think of neural network weights as a parameter. Hyperparameters or tuning parameters are meta parameters that influence the fitting process itself. For logistic regression, there are many hyperparameters like regularization strength C. For a neural network, there are many hyperparameters like the number of hidden layers. If all of this sounds confusing, [Jason Brownlee has a good rule of thumb](https://machinelearningmastery.com/difference-between-a-parameter-and-a-hyperparameter/) which is “If you have to specify a model parameter manually then it is probably a model hyperparameter.” "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<b> Hyperparameter Tuning </b>\n",
    "\n",
    "There are a lot of different ways to hyperparameter tune a decision tree for regression. One way is to tune the max_depth hyperparameter. max_depth (hyperparameter) is not the same thing as depth (parameter of a decision tree). max_depth is a way to preprune a decision tree. In other words, if a tree is already as pure as possible at a depth, it will not continue to split. If this isn’t clear, I highly encourage you to check out my Understanding Decision Trees for Classification (Python) tutorial to see the difference between max_depth and depth. \n",
    "\n",
    "The code below outputs the accuracy for decision trees with different values for max_depth.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "max_depth_range = list(range(1, 25))\n",
    "# List to store the average RMSE for each value of max_depth:\n",
    "r2_list = []\n",
    "for depth in max_depth_range:\n",
    "    reg = DecisionTreeRegressor(max_depth = depth,\n",
    "                            random_state = 0)\n",
    "    reg.fit(X_train, y_train)   \n",
    "    \n",
    "    score = reg.score(X_test, y_test)\n",
    "    r2_list.append(score)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The graph below shows that the best model R² is when the hyperparameter max_depth is equal to 5. This process of selecting the best model (max_depth = 5 in this case) among many other candidate models (with different max_depth values in this case) is called model selection. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x504 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig, ax = plt.subplots(nrows = 1, ncols = 1,\n",
    "                       figsize = (10,7),\n",
    "                       facecolor = 'white');\n",
    "ax.plot(max_depth_range,\n",
    "       r2_list,\n",
    "       lw=2,\n",
    "       color='r')\n",
    "ax.set_xlim([1, max(max_depth_range)])\n",
    "ax.grid(True,\n",
    "       axis = 'both',\n",
    "       zorder = 0,\n",
    "       linestyle = ':',\n",
    "       color = 'k')\n",
    "ax.tick_params(labelsize = 18)\n",
    "ax.set_xlabel('max_depth', fontsize = 24)\n",
    "ax.set_ylabel('R^2', fontsize = 24)\n",
    "ax.set_title('Model Performance on Test Set', fontsize = 24)\n",
    "fig.tight_layout()\n",
    "#fig.savefig('images/Model_Performance.png', dpi = 300)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Note that the model above could have still been overfitted on the test set since the code  changed max_depth repeatedly to achieve the best model. In other words, knowledge of the test set could have leaked into the model as the code iterated through 24 different values for max_depth (the length of max_depth_range is 24). This would lessen the power of our evaluation metric R² as it would no longer be as strong an indicator of generalization performance. This is why in real life, we often have training, test, and validation sets when hyperparameter tuning. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2>The Bias-variance Tradeoff</h2>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In order to understand why max_depth of 5 was the “best model” for our data, take a look at the graph below which shows the model performance when tested on the training and test set. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "# List of values to try for max_depth:\n",
    "max_depth_range = list(range(1, 25))\n",
    "\n",
    "# List to store the average RMSE for each value of max_depth:\n",
    "r2_test_list = []\n",
    "\n",
    "r2_train_list = []\n",
    "\n",
    "for depth in max_depth_range:\n",
    "    \n",
    "    reg = DecisionTreeRegressor(max_depth = depth, \n",
    "                             random_state = 0)\n",
    "    reg.fit(X_train, y_train)    \n",
    "    \n",
    "    score = reg.score(X_test, y_test)\n",
    "    r2_test_list.append(score)\n",
    "    \n",
    "    # Bad practice: train and test the model on the same data\n",
    "    score = reg.score(X_train, y_train)\n",
    "    r2_train_list.append(score)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(1.0, 24.0)\n",
      "(0.2, 1.0)\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x504 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,7), facecolor = 'white');\n",
    "\n",
    "ax.plot(max_depth_range,\n",
    "        r2_train_list,\n",
    "        lw=2,\n",
    "        color='b',\n",
    "        label = 'Training')\n",
    "\n",
    "ax.plot(max_depth_range,\n",
    "        r2_test_list,\n",
    "        lw=2,\n",
    "        color='r',\n",
    "        label = 'Test')\n",
    "\n",
    "ax.set_xlim([1, max(max_depth_range)])\n",
    "ax.grid(True,\n",
    "        axis = 'both',\n",
    "        zorder = 0,\n",
    "        linestyle = ':',\n",
    "        color = 'k')\n",
    "ax.tick_params(labelsize = 18)\n",
    "ax.set_xlabel('max_depth', fontsize = 24)\n",
    "ax.set_ylabel('R^2', fontsize = 24)\n",
    "ax.set_ylim(.2,1)\n",
    "\n",
    "ax.legend(loc = 'center right', fontsize = 20, framealpha = 1)\n",
    "ax.annotate(\"Best Model\",\n",
    "            xy=(5, 0.5558073822490773), xycoords='data',\n",
    "            xytext=(5, 0.4), textcoords='data', size = 20,\n",
    "            arrowprops=dict(arrowstyle=\"->\",\n",
    "                            connectionstyle=\"arc3\",\n",
    "                            color  = 'black', \n",
    "                            lw =  2),\n",
    "            ha = 'center',\n",
    "            va = 'center',\n",
    "            bbox={'facecolor':'white', 'edgecolor':'none', 'pad':5}\n",
    "            )\n",
    "\n",
    "ax.set_title('Model Performance on Training vs Test Set', fontsize = 24)\n",
    "\n",
    "# Annotating by figure fraction for ease because i want it outside the plotting area. \n",
    "ax.annotate('High Bias',\n",
    "            xy=(.1, .032), xycoords='figure fraction', size = 12)\n",
    "\n",
    "ax.annotate('High Variance',\n",
    "            xy=(.82, .032), xycoords='figure fraction', size = 12)\n",
    "\n",
    "temp = ax.get_xlim()\n",
    "temp1 = ax.get_ylim()\n",
    "\n",
    "fig.tight_layout()\n",
    "#fig.savefig('images/max_depth_vs_R2_Best_Model.png', dpi = 300)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Naturally, the training R² is always better than the test R² for every point on this graph because models make predictions on data they have seen before. \n",
    "\n",
    "To the left side of the “Best Model” on the graph (anything less than max_depth = 5), we have models that underfit the data and are considered high bias because they do not not have enough complexity to learn enough about the data. \n",
    "\n",
    "To the right side of the “Best Model” on the graph (anything more than max_depth = 5), we have models that overfit the data and are considered high variance because they are overly complex models that perform well on the training data, but perform badly on testing data. \n",
    "\n",
    "The “Best Model” is formed by minimizing bias error (bad assumptions in the model) and variance error (oversensitivity to small fluctuations/noise in the training set). \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2> Conclusion </h2>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![](images/grid_search_cross_validation.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "A goal of supervised learning is to build a model that performs well on new data which train test split helps you simulate. With any model validation procedure it is important to keep in mind some advantages and disadvantages which in the case of train test split are: \n",
    "\n",
    "Some Advantages: \n",
    "* Relatively simple and easier to understand than other methods like K-fold cross validation\n",
    "* Helps avoid overly complex models that don’t generalize well to new data\n",
    "\n",
    "Some Disadvantages: \n",
    "* Eliminates data that could have been used for training a machine learning model (testing data isn’t used for training) \n",
    "* Results can vary for a particular train test split (random_state)\n",
    "* When hyperparameter tuning, knowledge of the test set can leak into the model (this can be partially solved by using a training, test, and validation set). \n",
    "\n",
    "Future tutorials will cover other model validation procedures like K-fold cross validation ([pictured in the image above from the scikit-learn documentation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-evaluating-estimator-performance)) which help mitigate these issues. It is also important to note that [recent progress in machine learning has challenged the bias variance tradeoff](https://arxiv.org/abs/2109.02355) which is fundamental to the rationale for the train test split process.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![](images/DoubleDescentTestErrors.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "If you have any questions or thoughts on the tutorial, feel free to reach out on [Twitter](https://twitter.com/GalarnykMichael)."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
