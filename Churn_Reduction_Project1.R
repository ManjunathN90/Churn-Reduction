rm(list=ls(all=T))

getwd()
setwd("C:/Users/nagendrm/Desktop/Edwisor/Project/Churn Reduction/Result")
#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')

#install.packages(x)
lapply(x, require, character.only = TRUE)
rm(x)

## Read the Data
train = read.csv("Train_data.csv", header = T, na.strings = c(" ", "", "NA"))
test = read.csv("Test_data.csv", header = T, na.strings = c(" ", "", "NA"))

train_test = rbind.data.frame(train,test)

#Exploratory Data Anlysis

names(train)

str(train_test)

#Check number of unique values in each column
for(i in 1:ncol(train_test))
{
  print(colnames(train_test[i]))
  print(length(unique(train_test[,i])))
}

##Converting from numeric to factors which has small number of unique values
train_test$area.code = as.factor(as.character(train_test$area.code))
train_test$number.vmail.messages = as.factor(as.character(train_test$number.vmail.messages))
train_test$number.customer.service.calls = as.factor(as.character(train_test$number.customer.service.calls))

#No Missing Values
sum(is.na(train))
sum(is.na(test))


#convert string categories into factor numeric
#train dataset
for(i in 1:ncol(train_test)){
  
  if(class(train_test[,i]) == 'factor'){
    
    train_test[,i] = factor(train_test[,i], labels=(1:length(levels(factor(train_test[,i])))))
    
  }
}

#Since phone number has all the unique values and won't add much value to target variable, we will drop that column
train_test = subset(train_test, select = -c(phone.number))




############ Outlier  Analysis ###################
numeric_index = sapply(train_test, is.numeric)

numeric_data = train_test[, numeric_index]

cnames = colnames(numeric_data)


for (i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "Churn"), data = subset(train_test))+
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames[i],x="Churn")+
           ggtitle(paste("Box plot of Churn for",cnames[i])))
}

## Plotting plots together
gridExtra::grid.arrange(gn1,gn3,gn16,ncol=3)
gridExtra::grid.arrange(gn4, gn5,gn6,ncol=3)
gridExtra::grid.arrange(gn7,gn8,gn9,ncol=3)
gridExtra::grid.arrange(gn10,gn11,gn12,ncol=3)
gridExtra::grid.arrange(gn13,gn14,gn15,ncol=3)

#rm(gn1,gn2,gn3,gn4,gn5,gn6,gn7,gn8,gn9,gn10,gn11,gn12,gn13,gn14,gn15,gn16)

##Since we have outliers in all the continious variables, let's replace all outliers with Na and impute
##train dataset
for(i in cnames)
{
  val = train_test[,i][train_test[,i] %in% boxplot.stats(train_test[,i])$out]
  train_test[,i][train_test[,i] %in% val] = NA
  
}
train_test = knnImputation(train_test, k = 3)

############# Feature Selection ######################
##Correlation plot
corrgram(train_test[,numeric_index], order = F, upper.panel = panel.pie, text.panel = panel.txt, main = "Correlation Plot")

## Chi_Sqaured Test of Independence
factor_index = sapply(train_test, is.factor)
factor_data = train_test[,factor_index]

for (i in 1:length(colnames(factor_data)))
{
  print(names(factor_data)[i])
  print(chisq.test(table(factor_data$Churn,factor_data[,i])))
}

##Dimensionality Reduction
train_test = subset(train_test, select = -c(total.day.charge, total.eve.charge, total.night.charge, total.intl.charge, area.code ))

############# Feature Scaling ##########################
qqnorm(train_test$account.length)
hist(train_test$total.day.minutes)

train_test$state = as.numeric(train_test$state)
train_test$number.vmail.messages = as.numeric(train_test$number.vmail.messages)
train_test$number.customer.service.calls = as.numeric(train_test$number.customer.service.calls)


cnames = c("state","account.length", "number.vmail.messages","total.day.minutes", "total.day.calls","total.eve.minutes", "total.eve.calls", "total.night.minutes",  "total.night.calls",
           "total.intl.minutes", "total.intl.calls","number.customer.service.calls" )

#df = train_test

for(i in cnames)
{
  print(i)
  train_test[,i] = (train_test[,i] - min(train_test[,i]))/ (max(train_test[,i] - min(train_test[,i])))
}


############## Model Development ######################
rmExcept("train_test")

train = train_test[1:3333,]
test = train_test[3334:5000,]



######### Decision Tree for Classification ###########
c50_model = C5.0(Churn ~., train, trials = 100, rules = TRUE)

#Summary of DT model
summary(c50_model)


#write rules into disk
write(capture.output(summary(c50_model)), "c50Rules.txt")

#Predict for test cases
c50_Predictions = predict(c50_model, test[,-15], type = "class")

#Evaluate the performance of the model
confMatrix_c50 = table(test$Churn, c50_Predictions)
confusionMatrix(confMatrix_c50)

Result = test[,-15]
Result$Predicted_Churn_Result = c50_Predictions
#write.csv(Result,"output.csv", row.names = F)
#write(capture.output(confusionMatrix(confMatrix_c50)), "DT_result.txt")

#Accuracy : 95.5%
#FNR : 29.91%


######### Random Forest ##################

RF_model = randomForest(Churn ~., train, importance = TRUE, ntree = 500)

#predict
RF_Predictions = predict(RF_model, test[,-15])

#Evaluate
confMatrix_RF = table(test$Churn, RF_Predictions)
confusionMatrix(confMatrix_RF)


#Accuaracy: 94.84%
# FNR = 36.16%

########## Logistic Regression ##############
logit_model = glm(Churn ~., data = train, family = "binomial")

#Summary of the model
summary(logit_model)

#predict
logit_predictions = predict(logit_model, newdata = test, type = "response")


#Convert prob
logit_predictions = ifelse(logit_predictions > 0.5, 1, 0)


#Evalute
confMatrix_LR = table(test$Churn, logit_predictions)
confusionMatrix(confMatrix_LR)

#Accuracy: 87.52%
#FNR: 80%

######### Knn ###############
library(class)

#predict test data
KNN_Predictions = knn(train[,1:14], test[,1:14], train$Churn, k=13)

#Evaluate
conf_matrix = table(KNN_Predictions, test$Churn)
confusionMatrix(conf_matrix)

#Accuracy
sum(diag(conf_matrix))/nrow(test)

#Accuracy: 88.48%
#FNR: 22.41%


############ Naive Bayes ####################
library(e1071)

NB_model = naiveBayes(Churn ~., data = train)

#predict
NB_Predictions = predict(NB_model, test[, 1:14], type = 'class')

#confusion matrix
NB_conf_matrix = table(observed = test[,15], predicted = NB_Predictions)
confusionMatrix(NB_conf_matrix)

#Accuracy: 87.34%
#FNR: 77.67%