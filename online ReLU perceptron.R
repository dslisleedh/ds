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
#online learning
# a = learning rate
# x = input data
# y = label
# i = max iteration
# e_max = max error


setting_w_and_b <- function(a,x,y,i,e_max){
  cl <- makeCluster(detectCores())
  registerDoParallel(cl)
  y_hat <- c(rnorm(nrow(x)))
  p <- nrow(x)
  w <- round(abs(rnorm(ncol(x))), 3)
  b <- round(abs(rnorm(1)), 3)
  w_update <- data.frame(matrix(nrow = nrow(x), ncol = ncol(x)))
  b_update <- data.frame(matrix(nrow = nrow(x), ncol = 1))
  n_iteration <- 1
  max_iteration <- i
  y_hat <- 9999
  e <- 9999
  
  while(e > e_max && n_iteration <= max_iteration){
    for(i in 1:p){
      y_hat[i] <- ReLU(sum(x[i,]*w)+b)
      w <- w - (a * (y_hat[i] - y[i]) * stepf(sum(x[i,]*w)+b) * x[i,])
      b <- b - (a * (y_hat[i] - y[i]) * stepf(sum(x[i,]*w)+b))
    }
    e <- 1/length(y_hat) * 1/2 * sum((y_hat - y)^2)
    n_iteration <- n_iteration + 1
  }
  stopCluster(cl)
  return(list(n_iteration-1,w,b))
}


result <- setting_w_and_b(0.1,x,y,10000,0.0001)
result

for(i in 1:nrow(x)){
  print(ReLU(sum(x[i,] * result[[2]]) + result[[3]]))
}


