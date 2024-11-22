# Load the data

# Training data
letter_train <- read.table("Data/letter-train.txt", header = F, colClasses = "numeric")
Y <- letter_train[, 1]
X <- as.matrix(letter_train[, -1])

# Update training to set last part as validation
id_val = 1801:2000
Yval = Y[id_val]
Xval = X[id_val, ]
Ytrain = Y[-id_val]
Xtrain = X[-id_val, ]

# Testing data
letter_test <- read.table("Data/letter-test.txt", header = F, colClasses = "numeric")
Yt <- letter_test[, 1]
Xt <- as.matrix(letter_test[, -1])

# Source the NN function
source("FunctionsNN.R")

# [ToDo] Source the functions from HW3 (replace FunctionsLR.R with your working code)
source("FunctionsLR.R")

# Recall the results of linear classifier from HW3
# Add intercept column
Xinter <- cbind(rep(1, nrow(Xtrain)), Xtrain)
Xtinter <- cbind(rep(1, nrow(Xt)), Xt)

#  Apply LR (note that here lambda is not on the same scale as in NN due to scaling by training size)
out <- LRMultiClass(Xinter, Ytrain, Xtinter, Yt, lambda = 1, numIter = 150, eta = 0.1)
plot(out$objective, type = 'o')
plot(out$error_train, type = 'o') # around 19.5 if keep training
plot(out$error_test, type = 'o') # around 25 if keep training


# Apply neural network training with default given parameters
out2 = NN_train(Xtrain, Ytrain, Xval, Yval, lambda = 0.001,
                rate = 0.1, mbatch = 50, nEpoch = 150,
                hidden_p = 100, scale = 1e-3, seed = 12345)
plot(1:length(out2$error), out2$error, ylim = c(0, 70))
lines(1:length(out2$error_val), out2$error_val, col = "red")

# Evaluate error on testing data
test_error = evaluate_error(Xt, Yt, out2$params$W1, out2$params$b1, out2$params$W2, out2$params$b2)
test_error # 16.1

# Calculate two layer neural network running time
library(microbenchmark)
microbenchmark(
  NN_train(Xtrain, Ytrain, Xval, Yval, lambda = 0.001,
           rate = 0.1, mbatch = 50, nEpoch = 150,
           hidden_p = 100, scale = 1e-3, seed = 12345),
  times = 20
)

# [ToDo] Try changing the parameters above to obtain a better performance,
# this will likely take several trials

# First attempt - Small adjustments to baseline configuration
out1 = NN_train(Xtrain, Ytrain, Xval, Yval, 
                lambda = 0.0005,    # Reduce regularization
                rate = 0.1, 
                mbatch = 50, 
                nEpoch = 150,
                hidden_p = 200,     # Increase hidden layer nodes
                scale = 1e-3, 
                seed = 12345)

plot(1:length(out1$error), out1$error, ylim = c(0, 70))
lines(1:length(out1$error_val), out1$error_val, col = "red")
test_error1 = evaluate_error(Xt, Yt, out1$params$W1, out1$params$b1, out1$params$W2, out1$params$b2)
print(paste("Test Error 1:", test_error1))

# Second attempt - Larger network
out2 = NN_train(Xtrain, Ytrain, Xval, Yval, 
                lambda = 0.0003,    # Further reduce regularization
                rate = 0.1, 
                mbatch = 50, 
                nEpoch = 200,      # Increase training epochs
                hidden_p = 400,    # Significantly increase hidden layer nodes
                scale = 1e-3, 
                seed = 12345)

plot(1:length(out2$error), out2$error, ylim = c(0, 70))
lines(1:length(out2$error_val), out2$error_val, col = "red")
test_error2 = evaluate_error(Xt, Yt, out2$params$W1, out2$params$b1, out2$params$W2, out2$params$b2)
print(paste("Test Error 2:", test_error2))

# Third attempt - Deeper training strategy
out3 = NN_train(Xtrain, Ytrain, Xval, Yval, 
                lambda = 0.0001,   # Continue reducing regularization
                rate = 0.08,       # Reduce learning rate
                mbatch = 40,       # Reduce batch size
                nEpoch = 300,      # Further increase training epochs
                hidden_p = 600,    # Continue increasing hidden layer nodes
                scale = 5e-4,      # Reduce initial weights
                seed = 12345)

plot(1:length(out3$error), out3$error, ylim = c(0, 70))
lines(1:length(out3$error_val), out3$error_val, col = "red")
test_error3 = evaluate_error(Xt, Yt, out3$params$W1, out3$params$b1, out3$params$W2, out3$params$b2)
print(paste("Test Error 3:", test_error3))

# Final attempt - Extreme configuration
out4 = NN_train(Xtrain, Ytrain, Xval, Yval, 
                lambda = 0.00005,  # Very small regularization
                rate = 0.05,       # Smaller learning rate for stability
                mbatch = 30,       # Smaller batch size
                nEpoch = 400,      # More training epochs
                hidden_p = 1000,   # Very large hidden layer
                scale = 1e-4,      # Smaller initial weights
                seed = 12345)

plot(1:length(out4$error), out4$error, ylim = c(0, 70))
lines(1:length(out4$error_val), out4$error_val, col = "red")
test_error4 = evaluate_error(Xt, Yt, out4$params$W1, out4$params$b1, out4$params$W2, out4$params$b2)
print(paste("Test Error 4:", test_error4))

# Compare all results
results = data.frame(
  Configuration = c("Original", "Config 1", "Config 2", "Config 3", "Config 4"),
  Test_Error = c(16.1, test_error1, test_error2, test_error3, test_error4)
)
print(results)