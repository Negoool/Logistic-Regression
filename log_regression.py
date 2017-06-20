'''  Logistic Regression'''
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy.sparse import csr_matrix, hstack
import copy
import matplotlib.pyplot as plt

import os
os.system('cls')

# load data
data = pd.read_csv('amazon_baby_subset.csv')

# handle missing names or reviews by eliminating them
data = data.dropna(axis=0)
data.info()

# change the lable of class from -1 to 0
data.ix[data.sentiment == -1, 'sentiment'] = 0

# eliminate punctuation
data['new_column'] = data['review'].str.replace('[^\w\s]',' ')

def random_split(df, proportion):
    ''' split data frame randomly to two section based on the given proportion
    '''
    df_prim = copy.copy(df)
    df_shuffled = df.sample(frac = 1,  random_state =0)
    m = np.int (df_shuffled.shape[0]* proportion)
    df_train = df_shuffled[0:m]
    df_test = df_shuffled[m:]

    return df_train, df_test

df_train, df_test = random_split(data, .8)

# use scikit learn to count the frequency of words in each review
obj1 = CountVectorizer()
# a sparse matrix, bag of words
# train_matrix (i,j) = number of jth word in the ith review
train_matrix = obj1.fit_transform(df_train['new_column'])
#  the count matrix fot test set using the same word-index mapping
test_matrix = obj1.transform(df_test['new_column'])
# attribute of the obj, a dictionary, key:word, value: indice of that word
mapping = obj1.vocabulary_

# add the constant as the 0th feature to data
constant = csr_matrix(np.ones((train_matrix.shape[0],1)), dtype = int)
# add this column to the other features and change back the format to csr
# advantag of csr: good for arithmitic operation( in contrast to coo)
x_train = hstack([constant,train_matrix]).asformat('csr')
print x_train.shape
# same for test data
constant2 = csr_matrix(np.ones((test_matrix.shape[0],1)), dtype = int)
x_test = hstack([constant2,test_matrix]).asformat('csr')

def convert_to_np(df, target):
    ''' function to transform data frame to numpyarray'''
    y = df.as_matrix(target)
    return y
y_train = convert_to_np(df_train, ['sentiment'])
y_test = convert_to_np(df_test, ['sentiment'])


def prediction(teta , x):
    ''' compute h(x):the probability of being in class 1 for a given x and teta
    '''
    hx = 1./(1 + np.exp(-1*x.dot(teta)))
    return hx

def compute_j(hx, y):
    ''' compute the objective function
    '''
    N = y.shape[0]
    sum1 = np.sum(np.log(hx[y == 1]))
    sum2 = np.sum(np.log(1 - hx[y == 0]))
    J = (1./N)*(sum1+ sum2)
    return J

def logistic_regression(x_train, y_train, learning_rate, max_iter, initial_teta = None):
    ''' implement logistic regression using gradient descent
    the objevtive function of logistic regression is likelihood
    output : parameters and the amount of objective function VS iteration
    '''
    # number of data ponts in a training set
    N = x_train.shape[0]
    # dimention which is equal to total number of words in a word bag
    dimension = x_train.shape[1]
    # to store obj as iteration goes
    J_history = []
    # inititial value for parameters which is 0
    if initial_teta == None:
        initial_teta = np.zeros((dimension,1))
    teta = initial_teta[:]
    hx = prediction(teta , x_train)

    for iter in range(max_iter):
        # compute partial derevitive for current teta
        partial = (1./N)* ( x_train.transpose().dot((y_train - hx)) )
        # update teta
        teta = teta + (learning_rate*partial)
        # compute probability of being in class 1 for all data points
        hx = prediction(teta , x_train)
        # compute objective function
        J = compute_j(hx, y_train)
        J_history.append(J)

    return teta, J_history

def classification_error(teta, x, y):
    ''' compute classification error:number of misclassied/ total number of data
    '''
    hx = prediction(teta , x)
    hx[hx>=.5] = 1
    hx[hx<.5] = 0
    error = np.sum( y != hx)/float(x.shape[0])
    return error

## logistic regression for traing data
##finding appropraite learning rate &
## observing obj as iteration to notice it is non decresing
weights, J_history = logistic_regression(x_train, y_train, learning_rate =1e-1, max_iter = 1000, initial_teta = None)
plt.plot(range(1000), J_history)
print " classification error for train data", \
classification_error(weights, x_train, y_train)


show_result1 = 1
if show_result1 == 1:
    print
    # print the most positive and negative words
    indice_sorted_weights = np.argsort(weights,axis = 0)
    most_negative = indice_sorted_weights[0:10]
    most_positive = indice_sorted_weights[-1:-11:-1]

    print " most negative words"
    for indice in most_negative:
        for key, value in mapping.iteritems():
            # we sustract 1, because we added constant feature
            if value == (indice - 1):
                print key

    print " most positive words"
    for indice in most_positive:
        for key, value in mapping.iteritems():
            if value == (indice - 1):
                print key

show_result2 = 1
if show_result2 ==1:
    print
    hx = prediction(weights , x_train)
    indice_sorted_hx = np.argsort(hx,axis = 0)
    most_negative_reviews = indice_sorted_hx[0:5]
    most_positive_reviews = indice_sorted_hx[-1:-6:-1]

    print " most negative reviews"
    for indice in most_negative_reviews:
        print "founded probability", hx[indice]
        print "rating given by customer", df_train['rating'].iloc[indice]
        print df_train['review'].iloc[indice]

    print " most positive reviews"
    for indice in most_positive_reviews:
        print "founded probability", hx[indice]
        print "rating given by customer", df_train['rating'].iloc[indice]
        print df_train['review'].iloc[indice]

# assess the performence on test data
ll_train = compute_j(prediction(weights , x_train), y_train)
ll_test = compute_j(prediction(weights , x_test), y_test)
print " likihood for train and test data" ,ll_train, ll_test

error_train = classification_error(weights, x_train, y_train)
error_test = classification_error(weights, x_test, y_test)
print " classification errer for train and test data", error_train, error_test

def learning_curve(x_train, y_train, x_test, y_test, learning_rate =1e-1, max_iter = 5000, initial_teta = None, sectors =8):
    N = x_train.shape[0]
    j_train_test = []
    #for sec in range(sectors):
    for m_train in  xrange(100, N,5000):
        #m_train = np.int(((sec + 1.)/sectors)*N)
        print "*****",m_train
        x_train_sub = x_train[:m_train,:]
        y_train_sub = y_train[:m_train]
        weights, J_history = logistic_regression(x_train_sub, y_train_sub, learning_rate , max_iter,initial_teta  )
        ll_train = compute_j(prediction(weights , x_train_sub), y_train_sub)
        ll_test = compute_j(prediction(weights , x_test), y_test)
        j_train_test.append((ll_train, ll_test))
    return j_train_test

j_train_test = learning_curve(x_train, y_train, x_test, y_test, learning_rate = 1e-1, max_iter = 10000, initial_teta = None, sectors =8)
j_train = [-1*x[0] for x in j_train_test]
j_test = [-1*x[1] for x in j_train_test]
plt.figure
plt.plot( j_train, 'b')
plt.plot( j_test, 'r')
plt.show()
