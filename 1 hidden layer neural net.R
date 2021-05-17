x <- data.frame(rbind(c(0,0),
                      c(0,0),
                      c(0,0),
                      c(0,0),
                      c(0,1),
                      c(1,0),
                      c(0,1),
                      c(1,0),
                      c(0,1),
                      c(1,0),
                      c(0,1),
                      c(1,0),
                      c(1,1),
                      c(1,1),
                      c(1,1)))

y <- c(1,1,1,1,0,0,0,0,0,0,0,0,1,1,1)

ReLU <- function(x){
  return(max(0,x))
}
stepf <- function(x){
  if(x > 0){
    return(1)  
  }
  else{
    return(0)
  }
}

sigmoid <- function(x){
  return(1 / (1 + exp(-x)))
}


sigmoid_prime <- function(x){
  return(sigmoid(x) * (1 - sigmoid(x)))
}

#setting weights and bias of 1 hidden layer binary neural net
#activation function : ReLU function for hidden, sigmoid function for output
#loss function : MSE
#online learning
# a = learning rate
# x = input data
# y = label
# i_max = max iteration
# e_max = max error
# f = number btw 1~0 to prevent overfit
library(doParallel)
library(dplyr)
setting_w_b <- function(a,x,y,i_max,e_max,f){
  x <- as.matrix(x)
  cl <- makeCluster(detectCores())
  registerDoParallel(cl)
  n_neuron <- round(nrow(x) / (f * (ncol(x) + length(unique(y)))),0)
  w1 <- matrix(abs(rnorm(ncol(x) * n_neuron)),ncol = ncol(x), nrow = n_neuron)
  w2 <- abs(rnorm(n_neuron))
  b1 <- abs(rnorm(n_neuron))
  b2 <- abs(rnorm(1))
  n_iteration <- 1
  while(n_iteration < i_max){
    x1 <- foreach(i = 1:nrow(w1), .combine = 'cbind') %dopar% {
      as.matrix(x) %*% as.matrix(w1[i,]) + b1[i]
    }
    a1 <- matrix(mapply(ReLU, x1), nrow = nrow(x))
    a1_prime <- matrix(mapply(stepf, x1), nrow = nrow(x))
    x2 <- as.matrix(x1) %*% w2 + b2
    y_hat <- sigmoid(x2)
    e <- 1/length(y) * 1/2 * sum((y - y_hat)^2)
    if(e < e_max){
      break
    }else{
      for(i in 1:length(y)){
        w2 <- w2 - (a * (y_hat[i] - y[i]) * sigmoid_prime(x1[i,]*w2+b2) * a1[i,])
        b2 <- b2 - (a * (y_hat[i] - y[i]) * sigmoid_prime(sum(x1[i,]*w2+b2)))
      }
      for(i in 1:length(y)){
        w1 <- w1 - (a * a1_prime[i,] * (y_hat[i] - y[i] * w2) %*% t(x[i,]))
        b1 <- b1 - (a * a1_prime[i,] * (y_hat[i] - y[i] * w2))
      }
      n_iteration <- n_iteration + 1
    }
  }
  return(list(n_iteration-1,e,w1,w2,b1,b2))
}


result <- setting_w_b(0.1,x,y,10000,0.0001,1)
result

for(i in 1:nrow(x)){
  print(ReLU(sum(x[i,] * result[[2]]) + result[[3]]))
}



#
cl <- makeCluster(detectCores())
registerDoParallel(cl)
n_neuron <- round(nrow(x) / (1 * (ncol(x) + length(unique(y)))),0)
w1 <- matrix(abs(rnorm(ncol(x) * n_neuron)),ncol = ncol(x), nrow = n_neuron)
w2 <- abs(rnorm(n_neuron))
b1 <- abs(rnorm(n_neuron))
b2 <- abs(rnorm(1))
x1 <- foreach(i = 1:nrow(w1), .combine = 'cbind') %dopar% {
    as.matrix(x) %*% as.matrix(w1[i,]) + b1[i]
  }
a1 <- matrix(mapply(ReLU, x1), nrow = nrow(x))
a1_prime <- matrix(mapply(stepf, x1), nrow = nrow(x))
x2 <- as.matrix(x1) %*% w2 + b2
y_hat <- sigmoid(x2)
e <- 1/length(y) * 1/2 * sum((y - y_hat)^2)
if(e < e_max){
    break
  }else{
    for(i in 1:length(y)){
      w2 <- w2 - (a * (y_hat[i] - y[i]) * sigmoid_prime(x1[i,]*w2+b2) * a1[i,])
      b2 <- b2 - (a * (y_hat[i] - y[i]) * sigmoid_prime(sum(x1[i,]*w2+b2)))
    }
    for(i in 1:nrow(x))){
      w1 <- w1 - ((a * a1_prime[i,] * (y_hat[i] - y[i]) * w2  ) %*%  )
      b1 <- b1 - ()

    
      

a1
a1_prime
w2
w1
x[1,]
a1_prime[1,]
y_hat[1] - y[1]
w2
x[1,]
w1
a1
x[1,]

(0.1 * a1_prime[1,] * (y_hat[1] - y[1]) * w2)
x[9,]
(0.1 * a1_prime[1,] * (y_hat[1] - y[1]) * w2)
str(a)
x9 <- c(0,1)
a %*% t(as.matrix(x[9,]))
(0.1 * a1_prime[1,] * (y_hat[1] - y[1]) * w2) %*% t(x[9,])

a1_prime
