library(tidyverse)
library(ggplot2)
training_data <- read_csv("data/training_data_with_features_0-7-0-4_16May2024.csv") %>%
  mutate(
    nominal_time = sapply(lapply(strsplit(time, "/"), as.numeric), mean),
    sensor = ifelse(nominal_time >= 2013, "7/8/9", "5/7")
  )

p <- training_data %>%
  filter(code %in% c("land", "surf", "water", "clean_water", "noisy_water")) %>%
  ggplot(aes(x = swir16, y = nir08, colour = code)) +
  geom_point() +
  geom_smooth(method = "lm")


m <- lm(swir16 ~ nir08, data = training_data)
threshold <- predict(m, data.frame(nir08 = 0.128))
# 0.08099885
