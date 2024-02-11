library(tidyverse)
library(ggplot2)
training_data <- read_csv("data/training_data_with_features_1Feb2024.csv") %>%
  mutate(
    nominal_time = sapply(lapply(strsplit(training_data$time, "/"), as.numeric), mean),
    sensor = ifelse(nominal_time >= 2013, "7/8/9", "5/7")
  )

training_data %>%
  filter(code %in% c("land", "surf", "water", "clean_water", "noisy_water")) %>%
  ggplot(aes(x = swir16, y = nir08, colour = code)) +
  geom_point() +
  geom_smooth(method = "lm")
