# ==================================================
# 📊 Simple Linear Regression: Salary vs Experience
# Language: R
# Author: Chukwuka Chijioke Jerry
# ==================================================

# 📁 1. Load Required Libraries
library(ggplot2)
library(caTools)

# 📁 2. Import Dataset
dataset <- read.csv("../data/Salary_Data.csv")

# 🔀 3. Split the Dataset into Training and Test Sets
set.seed(1234)  # for reproducibility
split <- sample.split(dataset$Salary, SplitRatio = 0.8)
dataset_train <- subset(dataset, split == TRUE)
dataset_test <- subset(dataset, split == FALSE)

# 🧠 4. Fit Simple Linear Regression Model
regressor <- lm(Salary ~ YearsExperience, data = dataset_train)

# 📋 5. View Model Summary
summary(regressor)

# 📈 6. Predict Using the Model
predictions <- predict(regressor, newdata = dataset_test)

# 💾 7. Export Predictions to CSV
output <- data.frame(
  YearsExperience = dataset_test$YearsExperience,
  ActualSalary = dataset_test$Salary,
  PredictedSalary = predictions
)

# Create outputs directory if it doesn't exist
if (!dir.exists("../outputs")) dir.create("../outputs")
write.csv(output, "../outputs/SLR_predicted_salaries_r.csv", row.names = FALSE)

# 📊 8. Visualize the Training Set Results
plot1 <- ggplot(dataset_train, aes(x = YearsExperience, y = Salary)) +
  geom_point(color = "red", size = 3) +
  geom_smooth(method = "lm", se = FALSE, color = "blue") +
  xlab("Years of Experience") +
  ylab("Salary") +
  ggtitle("Salary vs Years of Experience (Training Set)") +
  theme_minimal()

# 📊 9. Visualize the Test Set Results
plot2 <- ggplot(dataset_test, aes(x = YearsExperience, y = Salary)) +
  geom_point(color = "red", size = 3) +
  geom_smooth(method = "lm", se = FALSE, color = "blue") +
  xlab("Years of Experience") +
  ylab("Salary") +
  ggtitle("Salary vs Years of Experience (Test Set)") +
  theme_minimal()

# Create plots directory if it doesn't exist
if (!dir.exists("../plots")) dir.create("../plots")
ggsave("../plots/R_linear_regression_plot_train.png", plot = plot1, width = 8, height = 5)
ggsave("../plots/R_linear_regression_plot_test.png", plot = plot2, width = 8, height = 5)

# ✅ 10. Final Status Message
cat("✅ Regression complete. Outputs saved in 'outputs/' and plots in 'plots/'.\n")
