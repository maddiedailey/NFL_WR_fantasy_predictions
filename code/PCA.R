#PCA Data Creation
df <- read.csv("/Users/noah/Desktop/CS229_Class/NFL_QB_fantasy_predictions/created_datasets/FINAL.csv")

# Assuming you have a dataset named 'df'
df <- df[df$Passing.Yards <= 400, ]
df <- df[df$Passing.Yards >= 100, ]

# Step 1: Separate 'Passing.Yards' column
passing_yards <- df$Passing.Yards
data <- df[, !colnames(df) %in% c("Passing.Yards")]

# Step 2: Perform PCA on the remaining columns
pca_result <- prcomp(data, scale. = TRUE)

# Step 3: Merge PCA results with 'Passing.Yards' column
pca_data <- data.frame(pca_result$x, Passing.Yards = passing_yards)
#pca_data <- pca_data[, !names(pca_data) %in%c("PC11", "PC12", "PC13","PC14","PC15","PC16","PC17",
#                          "PC18","PC19","PC20","PC21","PC22","PC23","PC24")]

#Step 4:
k <- 3
set.seed(123)
kmeans_result <- kmeans(pca_data[, -which(names(pca_data) == 'Passing.Yards')], centers = k)
#kmeans_result <- kmeans(pca_df, centers = k)
cluster_labels <- kmeans_result$cluster
# Add the cluster labels to your original dataset
pca_data$Cluster <- cluster_labels

#Step 5: Graph:
ggplot(pca_data, aes(x = as.factor(Cluster), y = Passing.Yards, group = as.factor(Cluster))) +
  geom_boxplot() + xlab("Cluster") + ylab("Passing Yards") + 
  ggtitle("Box Plot of Passing Yards by Cluster")

#Step 6: Save Centroid Values
# Get the centroid equations
centroid_equations <- kmeans_result$centers
# Save the centroid equations to a file
write.csv(centroid_equations, "/Users/noah/Desktop/CS229_Class/NFL_QB_fantasy_predictions/created_datasets/centroid_equations.csv", row.names = FALSE)

#Step 7: Compare Clusters
# Create empty lists for each cluster
cluster1_list <- list()
cluster2_list <- list()
cluster3_list <- list()

# Iterate over the rows of pca_data
for (i in 1:nrow(pca_data)) {
  cluster <- pca_data$Cluster[i]
  passing_yards <- pca_data$Passing.Yards[i]
  
  # Check the cluster value and add the passing yards to the respective list
  if (cluster == 1) {
    cluster1_list <- c(cluster1_list, passing_yards)
  } else if (cluster == 2) {
    cluster2_list <- c(cluster2_list, passing_yards)
  } else if (cluster == 3) {
    cluster3_list <- c(cluster3_list, passing_yards)

  }
}

#t-test
# Perform a t-test
t_test_result <- t.test(unlist(cluster1_list), unlist(cluster2_list))

# Print the test result
print(t_test_result)

# Perform a Wilcoxon rank sum test
wilcoxon_test_result <- wilcox.test(unlist(cluster1_list), unlist(cluster2_list))

# Print the test result
print(wilcoxon_test_result)

print(mean(unlist(cluster1_list)))
print(var(unlist(cluster1_list)))
print(mean(unlist(cluster2_list)))
print(var(unlist(cluster2_list)))
print(mean(unlist(cluster3_list)))

#Step 8: explain PCA
# Access the rotation (loadings) from PCA
loadings <- pca_result$rotation

# Print the loadings
print(loadings)
