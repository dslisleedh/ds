x <- data.frame(rbind(c(0,0),
                      c(0,1),
                      c(0,1),
                      c(0,0),
                      c(1,0),
                      c(1,1)))

y <- c(1,0,0,1,0,0)

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

#setting weights and bias of perceptron
#activation function : ReLU function
#loss function : MSE
#batch learning, parallel computing
# a = learning rate
# x = input data
# y = label
# i_max = max iteration
# e_max = max error

library(doParallel)

setting_w_and_b <- function(a,x,y,i_max,e_max){
  cl <- makeCluster(detectCores())
  registerDoParallel(cl)
  p <- nrow(x)
  w <- round(abs(rnorm(ncol(x))), 3)
  b <- round(abs(rnorm(1)), 3)
  w_update <- data.frame(matrix(nrow = nrow(x), ncol = ncol(x)))
  b_update <- data.frame(matrix(nrow = nrow(x), ncol = 1))
  n_iteration <- 1
  
  while(n_iteration <= i_max){
    net <- foreach(i = 1:nrow(x), .combine = 'append') %dopar% {
      sum(x[i,]*w + b)
    }
    y_hat <- foreach(i = 1:nrow(x), .combine = 'append', .export = 'ReLU') %dopar% {
      ReLU(net[i])
    }
    y_hat_prime <- foreach(i = 1:nrow(x), .combine = 'append', .export = 'stepf') %dopar% {
      stepf(net[i])
    }
    e <- (1/length(y)) * 1/2 * sum((y_hat - y)^2)
    if(e < e_max){
      break
    }else
    {
      w <- w - a * (1/length(y) * (sum(y_hat - y * x * y_hat_prime)))
      b <- b - a * (1/length(y) * (sum(y_hat - y)))
      n_iteration <- n_iteration + 1
    }
  }
  stopCluster(cl)
  return(list(n_iteration,w,b,e))
}

result <- setting_w_and_b(0.1,x,y,50000,0.001)
result

for(i in 1:nrow(x)){
  print(ReLU(sum(x[i,] * result[[2]]) + result[[3]]))
}

