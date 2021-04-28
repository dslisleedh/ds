x <- data.frame(rbind(c(0,0,0,0),
                      c(0,1,1,1),
                      c(0,1,0,1),
                      c(0,0,1,1),
                      c(1,0,1,0),
                      c(1,1,1,1)))

y <- c(1,0,0,0,0,0)

sigmoid <- function(x){
  return(1 / (1 + exp(-x)))
}

sigmoid_prime <- function(x){
  return(sigmoid(x) * (1 - sigmoid(x)))
}

#setting weights and bias of perceptron
#activation function : sigmoid function
#online learning 
# a = learning rate
# x = input data
# y = label
# i = max iteration
# e_max = max error


setting_w_and_b <- function(a,x,y,i,e_max){
  y_hat <- c()
  p <- nrow(x)
  w <- round(rnorm(ncol(x)), 3)
  b <- round(rnorm(1), 3)
  n_iteration <- 1
  max_iteration <- i
  y_hat <- 9999
  e <- 9999
  
  while(e > e_max && n_iteration <= max_iteration){
    for(i in 1:p){
      y_hat[i] <- sigmoid(sum(x[i,]*w)+b)
      w <- w - (a * (y_hat[i] - y[i]) * sigmoid_prime(sum(x[i,]*w)+b) * x[i,])
      b <- b - (a * (y_hat[i] - y[i]) * sigmoid_prime(sum(x[i,]*w)+b))
    }
    e <- 1/length(y_hat) * 1/2 * sum((y_hat - y)^2)
    n_iteration <- n_iteration + 1
  }
  return(list(n_iteration-1,w,b))
}


result <- setting_w_and_b(0.1,x,y,100000,0.001)
result


for(i in 1:nrow(x)){
  print(sigmoid(sum(x[i,] * result[[2]]) + result[[3]]))
}


