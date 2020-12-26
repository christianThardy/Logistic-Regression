# Logistic Regression

This timeless classification algorithm was trained on the same data used in this <a href="https://github.com/christianThardy/Feed-Forward-Artificial-Neural-Network" title="xtiandata.com" rel="nofollow">repo</a> to illustrate testing a range of network architectures to solve the same classification problem. The accuracy of the network in the aforementioned post was 88% while the accuracy of this logistic classifier is 92% approximately. Their architecture is similar: 

![2020-12-09](https://user-images.githubusercontent.com/29679899/101676964-e8477780-3a29-11eb-99f4-9de05ac1c28e.png)
### Logistic Regression

![2020-12-09](https://user-images.githubusercontent.com/29679899/101676996-f8f7ed80-3a29-11eb-911e-a48e2dc93247.gif)
### Feed-Forward Neural Network

They're both good predictors, they both provide a functional form f and parameter vector a to express `P(y|x)` as...
 
## `P(y|x) = f(x, a)`
 
The parameters of `a` are determined based on the data, usually by maximum likelihood estimation and the maximum likelihood estimation of the optimal parameter values a requires the maximization of `∏i=1nP(yi|xi,α)`. While they share similarities, the functional forms for logistic regression and neural network models are different. A network without a hidden layer is nothing more than a logistic regression model if a sigmoidal activation function is used. 
 
As `f` differs for logistic regression and FFNN's, the former is known as a parametric method, whereas the latter is sometimes called semi-parametric. This distinction is important because the contribution of parameters in logistic regression, or their coefficients & intercept, can be interpreted, and this is not always the case with the parameters of models that tend to have the highest predictive accuracy which are identified by the FFNN's weights.
 
In other words, NN's will give you an answer but they will not tell you why it arrived at its particular answer, not easily at least. This is still a very <a href="https://distill.pub/2019/activation-atlas/" title="distill.pub" rel="nofollow">active area of research</a>.
 
Understanding why your model arrived at a particular answer is important if you're trying to understand the underlying process for why a model arrived at its answer or if you need insights into how to improve the model etc. This trade-off is also related to the fact that you must represent your data as feature vectors for logistic regression, while neural networks are able to learn how to extract features by themselves. 
 
At the end of the day, the data will determine what type of architecture you should use. In terms of flexibility, a neural network's decision boundary is as state-of-the-art as universal approximation can get, but when you're facing hurdles like interpretability, computational expense, turnaround time, debugging or other various machine learning bottlenecks, the simplest solution is sometimes the best. 
 
Without verifying that the data used in a regression model meets a specific set of assumptions, the results of a given test could possibly be misleading. In a later post we'll discuss diagnosing the assumptions of logistic regression to properly validate the model and its statistical inferences. Some of these assumptions will shed light onto the linearity, normality, homoscedasticity and measurement level of logistic regression models.

</path></svg></a>Reference for regularized loss function and gradient:</h4>

![reference](https://user-images.githubusercontent.com/29679899/59174372-155a5780-8b1f-11e9-9e33-102b89d42816.png)
