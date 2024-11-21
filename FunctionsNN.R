# Initialization
#####################################################
# p - dimension of input layer
# hidden_p - dimension of hidden layer
# K - number of classes, dimension of output layer
# scale - magnitude for initialization of W_k (standard deviation of normal)
# seed - specified seed to use before random normal draws
initialize_bw <- function(p, hidden_p, K, scale = 1e-3, seed = 12345){
  # [ToDo] Initialize intercepts as zeros
  b1 = rep(0, hidden_p) 
  b2 = rep(0, K)
  
  # [ToDo] Initialize weights by drawing them iid from Normal
  # with mean zero and scale as sd
  set.seed(seed) 
  W1 = scale * matrix(rnorm(p * hidden_p), p, hidden_p)
  W2 = scale * matrix(rnorm(hidden_p * K), hidden_p, K) 
  
  # Return
  return(list(b1 = b1, b2 = b2, W1 = W1, W2 = W2))
}

# Function to calculate loss, error, and gradient strictly based on scores
# with lambda = 0
#############################################################
# scores - a matrix of size n by K of scores (output layer)
# y - a vector of size n of class labels, from 0 to K-1
# K - number of classes
loss_grad_scores <- function(y, scores, K){
  
  # [ToDo] Calculate loss when lambda = 0
  # loss = ...
  n = length(y)
  P = exp(scores) / rowSums(exp(scores))
  loss = - sum(log(P[cbind(1:n, y + 1)])) / n
  
  # [ToDo] Calculate misclassification error rate (%)
  # when predicting class labels using scores versus true y
  # error = ...
  predictions = max.col(P, ties.method = "first") - 1
  error = 100 * mean(predictions != y)
  
  # [ToDo] Calculate gradient of loss with respect to scores (output)
  # when lambda = 0
  # grad = ...
  P[cbind(1:n, y + 1)] = P[cbind(1:n, y + 1)] - 1
  grad = P / n 
  
  # Return loss, gradient and misclassification error on training (in %)
  return(list(loss = loss, grad = grad, error = error))
}

# One pass function
################################################
# X - a matrix of size n by p (input)
# y - a vector of size n of class labels, from 0 to K-1
# W1 - a p by h matrix of weights
# b1 - a vector of size h of intercepts
# W2 - a h by K matrix of weights
# b2 - a vector of size K of intercepts
# lambda - a non-negative scalar, ridge parameter for gradient calculations
one_pass <- function(X, y, K, W1, b1, W2, b2, lambda){

  # [To Do] Forward pass
  # From input to hidden 
  n = nrow(X)
  H = X %*% W1 + matrix(b1, n, length(b1), byrow = TRUE)
  # ReLU
  H = (abs(H) + H) / 2
  # From hidden to output scores
  scores = H %*% W2 + matrix(b2, n, K, byrow = TRUE)
  
  # [ToDo] Backward pass
  # Get loss, error, gradient at current scores using loss_grad_scores function
  out = loss_grad_scores(y, scores, K)
  # Get gradient for 2nd layer W2, b2 (use lambda as needed)
  dW2 = crossprod(H, out$grad) + lambda * W2
  db2 = colSums(out$grad)
  # Get gradient for hidden, and 1st layer W1, b1 (use lambda as needed)
  dH = tcrossprod(out$grad, W2)
  dH[H == 0] = 0  # Equivalent to the code: dH = dH * (A1 > 0) 
  dW1 = crossprod(X, dH) + lambda * W1
  db1 = colSums(dH)
  
  # Return output (loss and error from forward pass,
  # list of gradients from backward pass)
  return(list(loss = out$loss, error = out$error, grads = list(dW1 = dW1, db1 = db1, dW2 = dW2, db2 = db2)))
}

# Function to evaluate validation set error
####################################################
# Xval - a matrix of size nval by p (input)
# yval - a vector of size nval of class labels, from 0 to K-1
# W1 - a p by h matrix of weights
# b1 - a vector of size h of intercepts
# W2 - a h by K matrix of weights
# b2 - a vector of size K of intercepts
evaluate_error <- function(Xval, yval, W1, b1, W2, b2){
  # [ToDo] Forward pass to get scores on validation data
  nval <- length(yval)
  h    <- length(b1)
  K    <- length(b2)
  Hval = Xval %*% W1 + matrix(b1, nval, h, byrow = T)
  Hval = (abs(Hval) + Hval) / 2 
  scores_val = Hval %*% W2 + matrix(b2, nval, K, byrow = T)
  
  # [ToDo] Evaluate error rate (in %) when 
  # comparing scores-based predictions with true yval
  P = exp(scores_val) / rowSums(exp(scores_val))
  predictions = max.col(P, ties.method = "first") - 1
  error = 100 * mean(predictions != yval)
  
  return(error)
}


# Full training
################################################
# X - n by p training data
# y - a vector of size n of class labels, from 0 to K-1
# Xval - nval by p validation data
# yval - a vector of size nval of of class labels, from 0 to K-1, for validation data
# lambda - a non-negative scalar corresponding to ridge parameter
# rate - learning rate for gradient descent
# mbatch - size of the batch for SGD
# nEpoch - total number of epochs for training
# hidden_p - size of hidden layer
# scale - a scalar for weights initialization
# seed - for reproducibility of SGD and initialization
NN_train <- function(X, y, Xval, yval, lambda = 0.01,
                     rate = 0.01, mbatch = 20, nEpoch = 100,
                     hidden_p = 20, scale = 1e-3, seed = 12345){
  # Get sample size and total number of batches
  n = length(y)
  nBatch = floor(n/mbatch)

  # [ToDo] Initialize b1, b2, W1, W2 using initialize_bw with seed as seed,
  # and determine any necessary inputs from supplied ones
  initial.list = initialize_bw(ncol(X), hidden_p, length(unique(y)), scale, seed)
  W1 = initial.list$W1
  b1 = initial.list$b1
  W2 = initial.list$W2
  b2 = initial.list$b2
  
  # Initialize storage for error to monitor convergence
  error = rep(NA, nEpoch)
  error_val = rep(NA, nEpoch)
  
  # Set seed for reproducibility
  set.seed(seed)
  # Start iterations
  for (i in 1:nEpoch){
    # Allocate bathes
    batchids = sample(rep(1:nBatch, length.out = n), size = n)
    # Initialize current loss and current error
    cur_loss = 0 
    cur_error = 0
    # [ToDo] For each batch
    #  - do one_pass to determine current error and gradients
    #  - perform SGD step to update the weights and intercepts
    for (j in 1:nBatch){
      # Do one_pass to determine current error and gradients
      pass = one_pass(X[batchids == j, , drop = FALSE], y[batchids == j], length(unique(y)), W1, b1, W2, b2, lambda)
      gradient = pass$grads # get gradients
      
      # Accumulate loss and error
      cur_loss = cur_loss + pass$loss
      cur_error = cur_error + pass$error
      
      # Perform SGD step to update the weights and intercepts
      W1 = W1 - rate * gradient$dW1
      b1 = b1 - rate * gradient$db1
      W2 = W2 - rate * gradient$dW2
      b2 = b2 - rate * gradient$db2
    }
    
    # [ToDo] In the end of epoch, evaluate
    # - average training error across batches
    # - validation error using evaluate_error function
  }
  # Return end result
  return(list(error = error, error_val = error_val, params =  list(W1 = W1, b1 = b1, W2 = W2, b2 = b2)))
}