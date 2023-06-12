library(caret)
library(dplyr)
#Read in Data
df <- read.csv("/Users/noah/Desktop/wr.csv")

df <- na.omit(df)
df <- df[, -which(names(df) == c('Name','Week'))]
df <- df %>% filter(Receiving.Yards.x > 0)
df <- df %>% filter(avg_receiving_yards > 50)
df$rec_pc <- (df$avg_receiving_yards / (df$avg_receiving_yards + df$avg_teammate_rec_yards/16))


# Set the seed for reproducibility
set.seed(123)

# Generate train and test indices using createDataPartition()
train_indices <- createDataPartition(df$Receiving.Yards.x, p = 0.8, list = FALSE)

# Split the data into train and test
train_data <- df[train_indices, ]
test_data <- df[-train_indices, ]

model <- glm(Receiving.Yards.x ~ . - Receiving.TDs.x - total_receiving_yards_allowed, data=train_data,family=gaussian)
# Make predictions on the original dataset
predictions <- predict(model, newdata = test_data)
# Compare predicted vs. actual values
accuracy <- cor(test_data$Receiving.Yards.x, predictions)
cat("Correlation between predicted and actual values:", round(accuracy, 3))

#Provide a summary of model
summary(model)

# Load the ggplot2 package
library(ggplot2)

# Create a scatter plot of the predicted values against the actual values
ggplot(test_data, aes(x = Receiving.Yards.x, y = predictions)) + geom_point() + 
  geom_abline(intercept = 0, slope = 1, color = "red") + 
  labs(x = "Actual Yards", y = "Predicted Yards") + coord_fixed(ratio = 1) + 
  xlim(0, 150) + ylim(0, 150) + 
  labs(title="Predicted Yards for a Game (Using last 4 games)")

