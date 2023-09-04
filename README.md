# Final-Project_Analyzing-Data-with-R

# Install tidymodels if you haven't done so 
install.packages("rlang") 
install.packages("tidymodels")

# Library for modeling 
library(tidymodels)
# Load tidyverse 
library(tidyverse)

# Section 1. Download NOAA Weather Dataset

URL <- 'https://dax-cdn.cdn.appdomain.cloud/dax-noaa-weather-data-jfk-airport/1.1.4/noaa-weather-sample-data.tar.gz'
download.file(URL, destfile = 'noaa-weather-sample-data.tar.gz')
untar("noaa-weather-sample-data.tar.gz")

# Section 2.Extract and Read into Project

weather_sample_data <- read.csv("noaa-weather-sample-data/jfk_weather_sample.csv")
head(weather_sample_data)
glimpse(weather_sample_data)

# Section 3. Select Subset of Columns

selected_columns <- select(weather_sample_data, HOURLYRelativeHumidity, HOURLYDRYBULBTEMPF, 
                           HOURLYPrecip, HOURLYWindSpeed, HOURLYStationPressure)
head(selected_columns, 10)

# Section 4. Clean Up Columns
unique_precip <- unique(selected_columns$HOURLYPrecip) 
print(unique_precip)

modified_columns <- selected_columns %>%
  mutate(HOURLYPrecip = ifelse(HOURLYPrecip == "T", "0.0", HOURLYPrecip),
         HOURLYPrecip = str_remove(HOURLYPrecip, pattern = "s$"))

unique_precip_modified <- unique(modified_columns$HOURLYPrecip)
print(unique_precip_modified)

# Section 5. Convert Columns to Numerical Types
glimpse(modified_columns)
cleaned_data <- modified_columns %>% mutate(HOURLYPrecip = as.numeric(HOURLYPrecip))
glimpse(cleaned_data)

# Section 6. Rename Columns
final_data <- cleaned_data %>% rename( relative_humidity = HOURLYRelativeHumidity,
                                       dry_bulb_temp_f = HOURLYDRYBULBTEMPF, 
                                       precip = HOURLYPrecip, wind_speed = HOURLYWindSpeed, 
                                       station_pressure = HOURLYStationPressure ) 
glimpse(final_data)

# Section 7. Exploratory Data Analysis
set.seed(1234)
data_split <- initial_split(final_data, prop = 0.8)
train_data <- training(data_split)
test_data <- testing(data_split)

train_data %>%
  ggplot() +
  geom_histogram(aes(x = relative_humidity), binwidth = 1, fill = "blue", color = "black") +
  labs(title = "Histogram of Relative Humidity in Training Set") +
  xlab("Relative Humidity") +
  ylab("Frequency")

train_data %>%
  ggplot() +
  geom_histogram(aes(x = dry_bulb_temp_f), binwidth = 1, fill = "green", color = "black") +
  labs(title = "Histogram of Dry Bulb Temperature (F) in Training Set") +
  xlab("Dry Bulb Temperature (F)") +
  ylab("Frequency")

train_data %>%
  ggplot() +
  geom_histogram(aes(x = precip), binwidth = 0.01, fill = "purple", color = "black") +
  labs(title = "Histogram of Precipitation in Training Set") +
  xlab("Precipitation") +
  ylab("Frequency")

train_data %>%
  ggplot() +
  geom_histogram(aes(x = wind_speed), binwidth = 1, fill = "red", color = "black") +
  labs(title = "Histogram of Wind Speed in Training Set") +
  xlab("Wind Speed") +
  ylab("Frequency")

train_data %>%
  ggplot() +
  geom_histogram(aes(x = station_pressure), binwidth = 1, fill = "orange", color = "black") +
  labs(title = "Histogram of Station Pressure in Training Set") +
  xlab("Station Pressure") +
  ylab("Frequency")

train_data %>%
  pivot_longer(cols = c(relative_humidity, dry_bulb_temp_f, 
                        precip, wind_speed, station_pressure)) %>%
  ggplot(aes(x = name, y = value, fill = name)) +
  geom_boxplot() +
  labs(title = "Boxplots of Variables") +
  xlab("Variable") +
  ylab("Value") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Section 8 Linear Regression 
library(ggplot2)
model_relative_humidity <- lm(precip ~ relative_humidity, data = train_data)
ggplot(train_data, aes(x = relative_humidity, y = precip)) +
  geom_point() +
  stat_smooth(method = "lm", col = "red")

model_dry_bulb_temp_f <- lm(precip ~ dry_bulb_temp_f, data = train_data)
ggplot(train_data, aes(x = dry_bulb_temp_f, y = precip)) +
  geom_point() +
  stat_smooth(method = "lm", col = "green")

model_wind_speed <- lm(precip ~ wind_speed, data = train_data)
ggplot(train_data, aes(x = wind_speed, y = precip)) +
  geom_point() +
  stat_smooth(method = "lm", col = "purple")

model_station_pressure <- lm(precip ~ station_pressure, data = train_data)
ggplot(train_data, aes(x = station_pressure, y = precip)) +
  geom_point() +
  stat_smooth(method = "lm", col = "blue")

# Section 9. Improve the Model
model_multiple <- lm(precip ~ relative_humidity + dry_bulb_temp_f + wind_speed 
                     + station_pressure, data = train_data)

predictions_multiple <- predict(model_multiple, newdata = train_data, interval = "confidence")
mse_multiple <- mean(train_data$precip^2)
rmse_multiple <- sqrt(mse_multiple)
rsquared_multiple <- summary(model_multiple)$r.squared
summary(model_multiple)

model_poly <- lm(precip ~ poly(train_data$relative_humidity, 2, raw = TRUE), data = train_data)

predictions_poly <- predict(model_poly, newdata = train_data, interval = "confidence")
mse_poly <- mean(train_data$precip^2)
rmse_poly <- sqrt(mse_poly)
rsquared_poly <- summary(model_poly)$r.squared
summary(model_poly)

# Section 10. Find Best Model

# Evaluate the multiple linear regression model on the testing set
predictions_multiple_test <- predict(model_multiple, new_data = test_data, interval = "confidence")
mse_multiple_test <- mean(test_data$precip^2)
rmse_multiple_test <- sqrt(mse_multiple_test)
rsquared_multiple_test <- summary(model_multiple)$r.squared
summary(predictions_multiple_test)


# Evaluate the polynomial regression model on the testing set
predictions_poly_test <- predict(model_poly, newdata = test_data, interval = "confidence")
mse_poly_test <- mean(test_data$precip^2)
rmse_poly_test <- sqrt(mse_poly_test)
rsquared_poly_test <- summary(model_poly)$r.squared
summary(predictions_poly_test)

# Create a data frame to store the results
model_names <- c("Multiple Predictor Model", "Polynomial Model")
train_error <- c(rmse_multiple, rmse_poly)
test_error <- c(rmse_multiple_test, rmse_poly_test)
comparison_df <- data.frame(Model = model_names, Train_RMSE = train_error,
                            Test_RMSE = test_error)

print(comparison_df)
