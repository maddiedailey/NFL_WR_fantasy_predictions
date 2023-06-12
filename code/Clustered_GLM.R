library(caret)
library(dplyr)

#df should be obtained through PCA, so that it's already filtered and 
#has cluster assignments
#df <- read.csv("/Users/noah/Desktop/wr.csv")

df$rec_pc <- (df$avg_receiving_yards / (df$avg_receiving_yards + df$avg_teammate_rec_yards/16))
train_indices <- createDataPartition(df$Receiving.Yards.x, p = 0.8, list = FALSE)

# Split the data into train and test
train_data <- df[train_indices, ]
test_data <- df[-train_indices, ]

# Set the seed for reproducibility
set.seed(123)
total_predictions <- vector()
total_values <- vector()

for (i in 1:5) {
  train_data_strat <- train_data %>% filter(Cluster == i)
  test_data_strat <- test_data %>% filter(Cluster == i) 
  
  model <- glm(Receiving.Yards.x ~ . -Cluster, data=train_data_strat,family=gaussian)
  predictions <- predict(model, newdata = test_data_strat)
  print(cor(predictions,test_data_strat$Receiving.Yards.x))
  total_predictions <- append(total_predictions,predictions)
  total_values <- append(total_values,test_data_strat$Receiving.Yards.x)
}

#Get the total R^2 value
accuracy <- cor(total_values, total_predictions)
cat("Correlation between predicted and actual values:", round(accuracy, 3))


#PLOTTING DATA
# Load the ggplot2 package
library(ggplot2)

# Create a data frame with the vectors
df <- data.frame(x = total_values, y = total_predictions)

# Create a scatter plot of the predicted values against the actual values
ggplot(df, aes(x = x, y = y)) + geom_point() + 
  geom_abline(intercept = 0, slope = 1, color = "red") + 
  labs(x = "Actual Yards", y = "Predicted Yards") + coord_fixed(ratio = 1) + 
  xlim(0, 150) + ylim(0, 150) + 
  labs(title="Predicted Yards for a Game (Using last 4 games)")

