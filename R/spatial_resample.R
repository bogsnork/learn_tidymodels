#https://juliasilge.com/blog/map-challenge/

# Spatial resampling for #TidyTuesday and the #30DayMapChallenge
# Nov 5, 2021  

# install.packages("spDataLarge", repos = "https://nowosad.r-universe.dev")

library(tidyverse)

data("lsl", package = "spDataLarge")
landslides <- as_tibble(lsl)
landslides


# visualise

ggplot(landslides, aes(x, y)) +
  stat_summary_hex(aes(z = elev), alpha = 0.6, bins = 12) +
  geom_point(aes(color = lslpts), alpha = 0.7) +
  coord_fixed() +
  scale_fill_viridis_c() +
  scale_color_manual(values = c("gray90", "midnightblue")) +
  labs(fill = "Elevation", color = "Landslide?")

# Create spatial resamples

library(tidymodels)
library(spatialsample)

set.seed(123)
good_folds <- spatial_clustering_cv(landslides, coords = c("x", "y"), v = 5)
good_folds

# compare to non-spatial resamples

set.seed(234)
bad_folds <- vfold_cv(landslides, v = 5, strata = lslpts)
bad_folds


# plot the different resamples

plot_splits <- function(split) {
  p <- bind_rows(
    analysis(split) %>%
      mutate(analysis = "Analysis"),
    assessment(split) %>%
      mutate(analysis = "Assessment")
  ) %>%
    ggplot(aes(x, y, color = analysis)) +
    geom_point(size = 1.5, alpha = 0.8) +
    coord_fixed() +
    labs(color = NULL)
  print(p)
}


walk(good_folds$splits, plot_splits)
walk(bad_folds$splits, plot_splits)

# ok, so now we now spatial resampling is a good thing

# Fit and evaluate model ----

glm_spec <- logistic_reg()
lsl_form <- lslpts ~ slope + cplan + cprof + elev + log10_carea

lsl_wf <- workflow(lsl_form, glm_spec)

# doParallel::registerDoParallel()
set.seed(2021)
regular_rs <- fit_resamples(lsl_wf, bad_folds)
set.seed(2021)
spatial_rs <- fit_resamples(lsl_wf, good_folds)


# evaluate ----

collect_metrics(regular_rs)
collect_metrics(spatial_rs)

#> If we use the “regular” resampling, we get a more optimistc estimate of
#performance which would fool us into thinking our model would perform better
#than it really could. The lower performance estimate using spatial resampling
#is more accurate because of the autocorrelation of this geographic data;
#observations near each other are more alike than observations far apart. With
#geographic data, it’s important to use an appropriate model evaluation
#strategy!










