library(ggplot2)
library(car)
library(carData)
library(MASS)
library(lfactors)
library(Dire)
library(EdSurvey)


library(GGally)
#ggpairs(red_eclsk)
#Necessary Lib
#for (i in colnames(red_eclsk)) {
#  if (i != "C6R4MSCL") {  # Replace "y" with the actual name of your dependent variable
#    # Create the plot
#    plot(red_eclsk[[i]],red_eclsk$C6R4MSCL, main = paste("Scatter plot of", i, "vs y"), xlab = i, ylab = "y")
#}
#}


load("Project Data.RData")
red_eclsk <- eclsk_c
eclsk <- subset(eclsk_c,select = -CHILDID)

##Deleting irrelavant ID and binary variables that interfere.
#Deleting 
red_eclsk <-subset(eclsk_c,select = -c(1:2,34:36))
red_eclsk <-subset(red_eclsk, P1DISABL == 0)
red_eclsk <-subset(red_eclsk, F5SPECS == 0)
red_eclsk <-subset(red_eclsk,select = -c(33))
red_eclsk <-subset(red_eclsk,select = -c(F5SPECS))
red_eclsk <-subset(red_eclsk, P1DISABL == 0)
red_eclsk <-subset(red_eclsk, select = -c(P1DISABL))
red_eclsk <-subset(red_eclsk, GENDER == 1)
red_eclsk <-subset(red_eclsk, select = -c(GENDER))
red_eclsk <-subset(red_eclsk, WKWHITE == 1)
red_eclsk <-subset(red_eclsk, chg14 == 0)
red_eclsk <-subset(red_eclsk, P1FSTAMP == 0)
red_eclsk <-subset(red_eclsk, P1HSEVER == 0)
red_eclsk <-subset(red_eclsk, ONEPARENT == 0)

red_eclsk <-subset(red_eclsk, select = -c(WKWHITE,STEPPARENT,chg14,P1FSTAMP,P1HSEVER,ONEPARENT))
rej_eclsk <-subset(eclsk_c,select = c(33:37))

red_eclsk<-subset(red_eclsk, RIRT != 119.020)
#parameter of choice: Should we use the math results from first three grades?


cor_matrix <- cor(red_eclsk)
high_corr_df <- which(abs(cor_matrix) > 0.7 & cor_matrix != 1, arr.ind = TRUE)
names(high_corr_df) <- c("Variable1", "Variable2")

high_corr_values <- cor_matrix[high_corr_df]
# Combine the indices and values
high_corr_combined <- cbind(high_corr_df, Correlation = high_corr_values)
high_corr_combined
#Suspect collinear var given above



#Potential non-linear, dealt with later
plot(red_eclsk$MIRT,red_eclsk$C6R4MSCL)
poly_model <- lm(C6R4MSCL ~ poly(MIRT, 2), data = red_eclsk)  # Quadratic model
summary(poly_model)
red_eclsk<-data.frame(scale(red_eclsk,center = FALSE,scale=TRUE)) # Here the false is for BoxCox
#wHICH Turned out to be unnecessary, see normality

summary(poly_model)
pr_analysis<-prcomp(red_eclsk,scale=TRUE)

screeplot(pr_analysis)
summary(pr_analysis)

pc1_loadings<-pr_analysis$rotation[,1]
pc1_loadings

sorted_pc1_loadings <- sort(abs(pc1_loadings), decreasing = TRUE)
sorted_pc1_loadings

pc2_loadings<-pr_analysis$rotation[,2]
pc2_loadings

sorted_pc2_loadings <- sort(abs(pc2_loadings), decreasing = TRUE)
sorted_pc2_loadings
#PCA
#Roughly estimate significant estimators
barplot(pc1_loadings, main = "Loadings for PC1")


model_ana<-lm(C6R4MSCL~., data = red_eclsk)
summary(model_ana)
vif_values <- vif(model_ana)
print(vif_values) # very large value implies linearity
design_matrix <- model.matrix(model_ana)
svd_values <- svd(design_matrix)
condition_index <- sqrt(max(svd_values$d)/svd_values$d)
condition_index #similar to vif values

residualPlot(model_ana)
par(mfrow=(c(2,2)))
plot(model_ana)
#Four plots QQ-plots <=> normality, expect to break up near tails
#Residue plot important for checking Markov conditions of BLUE

#boxcox_result <- boxcox(model_ana)
#lambda_optimal <- boxcox_result$x[which.max(boxcox_result$y)]
#Try it :D


#Some really rough outlier elimination...
red_eclsk<-subset(red_eclsk, RIRT != 119.020)
# Install and load the lmtest package
model_ana <- lm(C6R4MSCL~., data = red_eclsk)
#anova(model_ana,reduced_model)
summary(model_ana)

red_eclsk$eMIRT<-exp(red_eclsk$MIRT)
new_model <- lm(C6R4MSCL~., data = red_eclsk)
#anova(model_ana,reduced_model)
summary(new_model)
par(mfrow = c(2, 2))
plot(new_model)

crPlots(new_model)
#Note how MIRT^2 cancels with MIRT
crPlots(model_ana)
save(eclsk,eclsk_c,file = "Project Data.RData")
#eclsk_df <- readECLS_K2011(filename = "childK5p.dat",
 #                          layoutFilename = "ECLSK2011_K5PUF.sps",
  #                         forceReread = FALSE,
   #                        verbose = TRUE) 

