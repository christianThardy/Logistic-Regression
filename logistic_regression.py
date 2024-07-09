# Stdlib dependencies
import warnings
warnings.filterwarnings('ignore')

# Third party dependencies
import pandas as pd
from pandas.plotting import parallel_coordinates
import numpy as np 
from scipy import optimize as opt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns


# Hyperbolic tangent activation function 
def tanh(z):
	return np.sinh(z)/np.cosh(z)


# Regularized loss function with L1 norm
def L1_norm_loss_function(theta, X, y, _lambda = 0.1):
	
	e = y.size
	h = tanh(X.dot(theta))
	reg = (_lambda/(2*e)) * np.sum(theta**2)
	
	return (1/e) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) + reg


# Regularized gradient function for L1 norm
def L1_norm_gradient(theta, X, y, _lambda = 0.1):
	
	e, f = X.shape
	theta = theta.reshape((f, 1))
	y = y.reshape((e, 1))
	h = tanh(X.dot(theta))
	reg = _lambda * theta / e
	
	return ((1 / e) * X.T.dot(h - y)) + reg


# Find the optimal theta co-efficient using logistic regression
def log_ reg_optimal_theta(X, y, theta):
    result = opt.minimize(fun = L1_norm_loss_function, x0 = theta, args = (X,y),
                          method = 'TNC', jac = L1_norm_gradient)
    return result.x


# Import data
purch_hist = pd.read_csv('purchase_history.csv')
data = pd.read_csv('data.csv')
data.head()

''' Visualizing the data to understand the distribution and separability. 
    Data can be separated by a non-linear solution. A rescaled 
    logistic sigmoid function would be the obvious solution.'''
df = data
sns.regplot(x = 'Age', y = 'Purchased', data = df)

# Output of the data is not linearly seperable 
plt.show()
age = sns.FacetGrid(data, hue = 'Purchased', size = 5,
                    palette = 'Blues_r').map(plt.scatter,
                                             'EstimatedSalary', 'Age')
plt.legend(loc = 'lower_right');
plt.show

# Plot parallel coordinates to see relevant correlations between the features and the output
parallel_coordinates(data, class_column = 'User ID', cols = ['Age', 'Purchased', 'EstimatedSalary'])
plt.gca().legend_.remove()
plt.show()

# Data preprocessing
Purchased = [0, 1] # Define output classes
e = purch_hist.shape[0] # Number of examples
f = 2 # # Number of features
k = 2

# Extract and preprocess features from the dataset
X = np.ones((e, f + 1))
y = np.array((e, 1))
X[:, 1] = purch_hist['Age'].values
X[:, 2] = purch_hist['EstimatedSalary'].values

# y label
y = purch_hist['Purchased'].values

# Mean normalization of features
for j in range(f):
	X[:, j] = (X[:, j] - X[:, j].mean())
	
# Split data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11)
 
'''Plot hyperbolic tangent function. The tanh activation function is sigmoidal in shape,
   0 is more sharply defined and is more efficient than the logistic function because 
   larger derivatives means faster minimization of the loss function.'''
x = np.arange(-5,5,0.01)
y = (2/(1 + np.exp(-2*x))) - 1
plt.plot(x, y)
plt.show()

# Initialize theta for all classes
all_theta = np.zeros((k, f + 1))
i = 0

# Train the model for each class 
for purchase in Purchased:
	tmp_y = np.array(y_train == purchase, dtype = int)
	optTheta = log_ reg_optimal_theta(X_train, tmp_y, np.zeros((f + 1, 1)))
	all_theta[i] = optTheta
	i += 1

# Make predictions on the test set
P = tanh(X_test.dot(all_theta.T)) # Probability for each customers purchase decision
P = [Purchased[np.argmax(P[i, :])] for i in range(X_test.shape[0])]

# Calculate and print the accuracy of the model
print('Test Accuracy', accuracy_score(y_test, P) * 140, '%')
