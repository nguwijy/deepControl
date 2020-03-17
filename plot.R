rm(list = ls())

wd <- "C:/Users/nguwijy/Dropbox/Projects/blog/deep_control"
setwd(wd)

# merton problem...
filelist <- list.files(pattern = "control_log.txt")

datalist = lapply(filelist, function(x)read.table(x, sep=",", header=T)) 

datafr = do.call("rbind", datalist)

colors_val <- c("black", "red", "green", "blue", "pink", "yellow")
pch_val <- c(16, 18, 18, 18, 18, 18)
legend_val <- c("exact solution", "deep approximation")

resol <- 150

jpeg("merton.jpg", units="in", width=5, height=5, res=resol)


exact_sol <- exp(-(0.02/0.1)^2/2*0.1)*exp(-exp(0.03*0.1))
plot(datafr$iteration, datafr$loss, 
     main = "Merton Problem", xlab = "iteration", ylab = "value function",
     pch = pch_val[2], col = colors_val[2])
abline(h = exact_sol, col = colors_val[1])

legend("topright", legend = legend_val_M[1:2], 
       col = colors_val[1:2], pch = pch_val[1:2],
       cex=0.75)
dev.off()

# dummy problem...
filelist <- list.files(pattern = "control_log_dummy.txt")

datalist = lapply(filelist, function(x)read.table(x, sep=",", header=T)) 

datafr = do.call("rbind", datalist)
jpeg("dummy.jpg", units="in", width=5, height=5, res=resol)

exact_sol <- 0
plot(datafr$iteration, datafr$loss, 
     main = "Minimization Problem", xlab = "iteration", ylab = "value function",
     pch = pch_val[2], col = colors_val[2])
abline(h = exact_sol, col = colors_val[1])

legend("topright", legend = legend_val_M[1:2], 
       col = colors_val[1:2], pch = pch_val[1:2],
       cex=0.75)
dev.off()
