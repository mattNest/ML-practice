# Data Preprocessing

# Importing the dataset
data = read.csv("Data.csv")

# Taking care of missing data -- method1
data$Age = ifelse(is.na(data$Age),
                  ave(data$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                  data$Age)

data$Salary = ifelse(is.na(data$Salary),
                  ave(data$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                  data$Salary)

data_backup <-  data

# Taking care of missing data -- method2
data2 <- read.csv("Data.csv")

# Mean Imputation
age_mean <- mean(data2[,"Age"],na.rm = TRUE)
salary_mean <- mean(data2[,"Salary"],na.rm= TRUE)

data2[is.na(data2$Age),"Age"] <- age_mean
data2[is.na(data2$Salary),"Salary"] <- salary_mean


# Encoding catergorical data

# Encoding Country 
data2$Country = factor(data2$Country,
                      levels = c('France','Spain','Germany'),
                      labels = c(1,2,3))
summary(data2)
str(data2)

# Encoding Purchased
data2$Purchased = factor(data2$Purchased,
                         levels = c('No','Yes'),
                         labels = c(0,1))

summary(data2)
str(data2)

# Splitting the dataset into the Training Set and Test Set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(data2$Purchased, SplitRatio = 0.8)
training_set = subset(data2, split == TRUE)
test_set = subset(data2, split == FALSE)

# Feature Scaling 
training_set = scale(training_set[, 2:3])
test_set[, 2:3] = scale(test_set[, 2:3])


