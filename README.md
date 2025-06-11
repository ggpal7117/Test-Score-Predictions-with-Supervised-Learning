# Test-Score-Predictions-with-Supervised-Learning
Using a machine learning pipeline, I found an optimal model to predict student test scores (95% accuracy).

# Data Preprocessing and One-Hot Encoding
The first part of the project was exploring the dataset. Certain boolean values were initially set as "yes" and "no", so we had to change that. We also had to take into account missing values. After this, we needed to identify the categorical and numerical features to ultimately predict our target(test grade). Some features, such as "sleep," "tutoring," and "physical activity," are represented numerically but follow a normal distribution across a given range. These features need to be converted to a categorical form because of this. Using dummy variables, I could transform these features to categorical ones that the ML models could easily digest. 

# Model Selection and Pipeline
The five models decided were some of the most popular classification models in Machine Learning. Logistic Regression, Support Vector Machine(Classifier), K-Nearest Neighbors, Decision Trees, and SGD classifier.
The models were all fed different hyperparameters that were run through a randomized search(time-efficient method) to find the parameters and score for that specific model. All data was scaled for this process as well.

## Logistic Regression
![Image](https://github.com/user-attachments/assets/bf6e7a96-1bff-460e-bedd-f01117b0faa3)
Typically and most famously, logistic regression is used for binary classification using the sigmoid function. This outputs probabilities for both 0 and 1 classes. However, in our case, we are making predictions out of 5 different options(A-F). This means we will use a multiclass logistic regression. This model often uses a softmax function to compute probabilities for all classes.
![Image](https://github.com/user-attachments/assets/425aef39-2a0f-4603-8b33-496b3fa3f6df)


## KNN
The K-Nearest Neighbors algorithm is one of the most popular classification models. It compares a data point to K number of training points and simply assigns the most frequent class among them. In terms of multiclass classification, KNN naturally supports it by using neighbor voting classification. The below gives a good visual of how it works
![Image](https://github.com/user-attachments/assets/46b5ae5a-bca4-48e2-ab03-f2c133da1764)


## Support Vector Machines
Support Vector Machine Models are most commonly used for classification tasks as they work to find an optimal hyperplane to distinguish different categories. SVMs use kernels to find decision boundaries. In our model it uses a simple linear kernel to split data across a straight plane.
![Image](https://github.com/user-attachments/assets/f94e0ea1-76ac-445b-a0f1-bab09ef51ea7)


## Decision Trees
Decision Trees are a common model often used in other ensemble learning techniques. The model starts with the dataset at the root node, and then chooses important features to split the data into smaller subsets. This splitting process continues until a leaf node is reached, where a prediction is eventually made.
![Image](https://github.com/user-attachments/assets/02cc7e89-8e65-4a44-a978-67601a24560d)


## SDG Classifier
The final model used in the pipeline is the SDG Classifier. This model, depending on the loss function(hinge, log-loss), acts as an SVM/Logistic Regression. SDG classifiers are especially useful for large datasets, which ours is to an extent, at 6600 rows of data.

# Scoring/Testing
We then scored and tested the best model, which happened to be the simplest, the Logistic Regression. This model made nearly 98% correct predictions on the test data, showing the power of these models.

# VotingClassifier
To finish, to see if a combination of the models would improve performance even more, I fed in a KNN, SVC, Logreg, and DT model into Python's VotingClassifier. This ensemble learning method feeds uses multiple models to make predictions, then finally votes(soft votes) on predictions. 
![Image](https://github.com/user-attachments/assets/51a6b2e8-ef14-40ac-891b-af4d40b1c23b)

# Thank You
