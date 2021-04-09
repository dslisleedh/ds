#linear unit perceptron delta rule

x = data.frame(rbind(
  c(-1,-1,1),
  c(-1,1,1),
  c(1,-1,1),
  c(1,1,1)))
y = c(-1,1,1,1)

#d = data(dataframe)
#l = label(vector)
#a = learning rate
#t = threshold
delta_rule <- function(d,l,a,t){
  p <- nrow(d)
  epoch_initial_w <- round(rnorm(ncol(d)),2)
  epoch_end_w <- -epoch_initial_w
  n_iteration <- 0
  while(!identical(epoch_end_w,epoch_initial_w)){
    print(n_iteration)
    epoch_initial_w <- epoch_end_w
    w <- epoch_initial_w
    for(i in 1:p){
      net <- sum(x[i,] * w)
      ifelse(net > t, y_hat <- 1, ifelse(net == t, y_hat <- 0, y_hat <- -1) )
      if(y_hat != y[i]){
        w <- w + a*(y[i] - y_hat)*d[i,]
      }
    epoch_end_w <- w
    n_iteration <- n_iteration + 1
    }
  }
  return(as.numeric(epoch_end_w))
}
delta_rule(x,y,1,0)


as.matrix(x) %*% delta_rule(x,y,1,0)
