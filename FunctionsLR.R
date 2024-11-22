# Function that implements multi-class logistic regression.
#############################################################
# Description of supplied parameters:
# X - n x p training data, 1st column should be 1s to account for intercept
# y - a vector of size n of class labels, from 0 to K-1
# Xt - ntest x p testing data, 1st column should be 1s to account for intercept
# yt - a vector of size ntest of test class labels, from 0 to K-1
# numIter - number of FIXED iterations of the algorithm, default value is 50
# eta - learning rate, default value is 0.1
# lambda - ridge parameter, default value is 1
# beta_init - (optional) initial starting values of beta for the algorithm, should be p x K matrix 

## Return output
##########################################################################
# beta - p x K matrix of estimated beta values after numIter iterations
# error_train - (numIter + 1) length vector of training error % at each iteration (+ starting value)
# error_test - (numIter + 1) length vector of testing error % at each iteration (+ starting value)
# objective - (numIter + 1) length vector of objective values of the function that we are minimizing at each iteration (+ starting value)
LRMultiClass <- function(X, y, Xt, yt, numIter = 50, eta = 0.1, lambda = 1, beta_init = NULL){
  ## Check the supplied parameters as described. You can assume that X, Xt are matrices; y, yt are vectors; and numIter, eta, lambda are scalars. You can assume that beta_init is either NULL (default) or a matrix.
  ###################################
  #Define the variables' dimension
  n = nrow(X)
  p = ncol(X)
  K = length(unique(y)) 
  
  # Check that the first column of X and Xt are 1s, if not - display appropriate message and stop execution.
  if (!all(X[, 1] == 1)) {
    stop("First column of X should be 1s.") 
  }
  if (!all(Xt[, 1] == 1)) {
    stop("First column of Xt should be 1s.") 
  }
  # Check for compatibility of dimensions between X and Y
  if (length(y) != n) {
    stop("The number of rows of X should be the same as the length of y.") 
  }
  # Check for compatibility of dimensions between Xt and Yt
  if (length(yt) != nrow(Xt)) {
    stop("The number of rows of Xt should be the same as the length of yt.") 
  }
  # Check for compatibility of dimensions between X and Xt
  if (ncol(X) != ncol(Xt)) {
    stop("The number of columns in X should be the same as the number of columns in Xt.") 
  }
  # Check eta is positive
  if (eta <= 0) {
    stop("The learning rate eta should be positive.")
  }
  # Check lambda is non-negative
  if (lambda < 0) {
    stop("The ridge parameter lambda should be non-negative.")
  }
  # Check whether beta_init is NULL. If NULL, initialize beta with p x K matrix of zeroes. If not NULL, check for compatibility of dimensions with what has been already supplied.
  if (is.null(beta_init)) {
    beta <- matrix(0, p, K)
  } else {
    if (nrow(beta_init) != p | ncol(beta_init) != K) {
      stop("The dimensions of beta_init supplied are not correct.")
    }
    beta <- beta_init
  }
  
  #Define some useful functions
  # Helper function to calculate probabilities
  calc_probs <- function(X, beta) {
    exp_Xbeta <- exp(X %*% beta)
    probs <- exp_Xbeta / rowSums(exp_Xbeta)
    return(probs)
  }
  
  # Helper function to calculate objective
  calc_objective <- function(X, y, lambda, beta) {
    P = calc_probs(X, beta)
    return( - sum(log(P[cbind(1:nrow(X), y + 1)])) + 0.5 * lambda * sum(beta^2))
  }
  
  # Helper function to calculate error rate
  calc_error <- function(X, y, beta) {
    probs <- calc_probs(X, beta)
    predicted_class <- max.col(probs) - 1
    error_rate <- mean(predicted_class != y) * 100
    return(error_rate)
  }
  
  
  ## Calculate corresponding pk, objective value f(beta_init), training error and testing error given the starting point beta_init
  ##########################################################################
  error_train = numeric(numIter + 1)
  error_test = numeric(numIter + 1)
  objective = numeric(numIter + 1) 
  
  objective[1] = calc_objective(X, y, lambda, beta)
  error_train[1] = calc_error(X, y, beta)
  error_test[1] = calc_error(Xt, yt, beta)
  
  ## Newton's method cycle - implement the update EXACTLY numIter iterations
  ##########################################################################
 
  # Within one iteration: perform the update, calculate updated objective function and training/testing errors in %
  for (i in 1:numIter) { 
    P = calc_probs(X, beta)
    for (j in 1:K) { 
      P_k = P[, j]
      # computes gradient
      grad = t(X) %*% (P[, j] - (1 * (y == (j - 1)))) + lambda * beta[, j]
      # computes Hessian
      W = sqrt(P_k * (1 - P_k))
      H = t(X * W) %*% (X * W) + lambda * diag(p)
      beta[, j] = beta[, j] - eta * solve(H) %*% grad
      # updates beta_k according to the damped Newton's method
    }
    objective[i + 1] = calc_objective(X, y, lambda, beta)
    error_train[i + 1] = calc_error(X, y, beta)
    error_test[i + 1] = calc_error(Xt, yt, beta)
  }
  
  ## Return output
  ##########################################################################
  # beta - p x K matrix of estimated beta values after numIter iterations
  # error_train - (numIter + 1) length vector of training error % at each iteration (+ starting value)
  # error_test - (numIter + 1) length vector of testing error % at each iteration (+ starting value)
  # objective - (numIter + 1) length vector of objective values of the function that we are minimizing at each iteration (+ starting value)
  return(list(beta = beta, error_train = error_train, error_test = error_test, objective =  objective))
}