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
  cl <- makeCluster(detectCores())
  registerDoParallel(cl)
  n_neuron <- round(nrow(x) / (f * (ncol(x) + length(unique(y)))),0)
  w1 <- matrix(abs(rnorm(ncol(x) * n_neuron)),ncol = ncol(x), nrow = n_neuron)
  w2 <- abs(rnorm(n_neuron))
  b1 <- abs(rnorm(n_neuron))
  b2 <- abs(rnorm(1))
  while(n_iteration < max_iteration){
    x1 <- foreach(i = 1:nrow(w1), .combine = 'cbind') %dopar% {
      as.matrix(x) %*% as.matrix(w1[i,]) + b1[i]
    }
    a1 <- matrix(mapply(ReLU, x1), nrow = nrow(x))
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
        w1 <- w1 - (a * *stepf(x[i,]*w1 + b1) * x[i,])
        b1 <- b1 - ()
      }
    }
  }
}
w1
x
x1
x1[1,]
x1[1,]*w2
as.matrix(x) %*% w1[1,] + b1[1]

a1
x1
w2
a1
x2 %*% w2
a1[] <- lapply(x1, ReLU)
a1
#########
x1


f <- 1
n_neuron <- round(nrow(x) / (f * (ncol(x) + length(unique(y)))),0)
w1 <- matrix(abs(rnorm(ncol(x) * n_neuron)),ncol = ncol(x), nrow = n_neuron)
w2 <- abs(rnorm(n_neuron))
b1 <- abs(rnorm(n_neuron))
b2 <- abs(rnorm(1))
x1 %>% mutate_all(ReLU)

w1
b1
a1
w2
cl <- makeCluster(detectCores())
registerDoParallel(cl)
x1 <- foreach(i = 1:nrow(w1), .combine = 'cbind') %dopar% {
  as.matrix(x) %*% as.matrix(w1[i,]) + b1[i]
}
a1 <- matrix(mapply(ReLU, x1), nrow = nrow(x))
x2 <- as.matrix(x1) %*% w2 + b2
x2
y_hat <- sigmoid(x2)
e <- 1/length(y) * 1/2 * sum((y_hat - y)^2)
e
y_hat
