# =============================================================
# Coffee Shop Sales Forecasting — v2 (fair model comparison)
#
# What's different from v1:
#   1. Dates come from transaction_date (v1's as.Date(datetime)
#      converted to UTC and moved evening sales to the next day).
#   2. Every model is trained on Jan–May 2023 and tested on the
#      same unseen month (June 2023) — a true 30-day forecast.
#   3. A simple baseline ("same weekday last week") to beat.
#   4. The neural network uses only information known in advance
#      (trend + day of week), not same-day quantity, and is judged
#      over 10 runs because its results vary with random starts.
#   5. The final 30-day forecast uses the best model on June.
#   6. Day x hour heatmap for staffing.
# All charts display in the RStudio Plots tab (nothing is saved).
# =============================================================

library(dplyr)
library(ggplot2)
library(forecast)
library(neuralnet)

options(scipen = 999, digits = 4)

# ----- Load and prepare -----
# Put "Coffee Shop Sales.csv" in the data/ folder, or pick it when prompted
data_path <- "data/Coffee Shop Sales.csv"
if (!file.exists(data_path)) data_path <- file.choose()
coffee.df <- read.csv(data_path)

coffee.df$date        <- as.Date(coffee.df$transaction_date, format = "%m/%d/%Y")
coffee.df$datetime    <- as.POSIXct(paste(coffee.df$transaction_date, coffee.df$transaction_time),
                                    format = "%m/%d/%Y %H:%M:%S")
coffee.df$hour_of_day <- as.numeric(format(coffee.df$datetime, "%H"))
coffee.df$day_of_week <- factor(weekdays(coffee.df$date),
                                levels = c("Monday", "Tuesday", "Wednesday", "Thursday",
                                           "Friday", "Saturday", "Sunday"))

daily_sales <- coffee.df %>%
  group_by(date) %>%
  summarise(total_qty   = sum(transaction_qty),
            total_sales = sum(transaction_qty * unit_price),
            .groups = "drop") %>%
  arrange(date) %>%
  mutate(t           = row_number(),                       # day index (trend)
         day_of_week = factor(weekdays(date), levels = levels(coffee.df$day_of_week)))

cat("Days of data:", nrow(daily_sales), " (", format(min(daily_sales$date)), "to",
    format(max(daily_sales$date)), ")\n")

# ----- Time-based split: train Jan–May, test June -----
train <- daily_sales %>% filter(date <  as.Date("2023-06-01"))
test  <- daily_sales %>% filter(date >= as.Date("2023-06-01"))
h <- nrow(test)
cat("Train days:", nrow(train), " Test days (June):", h, "\n")

# Error metrics
score <- function(model, actual, predicted) {
  data.frame(model = model,
             MAE   = mean(abs(actual - predicted)),
             RMSE  = sqrt(mean((actual - predicted)^2)),
             MAPE  = mean(abs((actual - predicted) / actual)) * 100)
}

train_ts <- ts(train$total_sales, frequency = 7)

# ----- 1. Baseline: same weekday last week (seasonal naive) -----
pred_naive <- as.numeric(snaive(train_ts, h = h)$mean)

# ----- 2. Linear regression: trend + day of week -----
lm_model <- lm(total_sales ~ t + day_of_week, data = train)
cat("\nLinear Regression Model Summary:\n")
print(summary(lm_model))
pred_lm <- as.numeric(predict(lm_model, newdata = test))

# ----- 3. ARIMA with weekly seasonality -----
arima_fit <- auto.arima(train_ts)
cat("\nARIMA model:\n")
print(arima_fit)
pred_arima <- as.numeric(forecast(arima_fit, h = h)$mean)

# ----- 4. Neural network: trend + day of week (known in advance) -----
make_nn_inputs <- function(df) {
  x <- as.data.frame(model.matrix(~ day_of_week, data = df))[, -1]   # 6 weekday dummies
  names(x) <- gsub("day_of_week", "dow_", names(x))
  x$t_scaled <- (df$t - min(train$t)) / (max(train$t) - min(train$t))
  x
}
y_min <- min(train$total_sales); y_max <- max(train$total_sales)

nn_train <- make_nn_inputs(train)
nn_train$y <- (train$total_sales - y_min) / (y_max - y_min)
nn_test  <- make_nn_inputs(test)

nn_formula <- as.formula(paste("y ~", paste(setdiff(names(nn_train), "y"), collapse = " + ")))

# Neural networks start from random weights, so one run can be lucky or unlucky.
# Train it 10 times (seeds 1-10) and judge it on the average result.
nn_runs <- lapply(1:10, function(s) {
  set.seed(s)
  m <- neuralnet(nn_formula, data = nn_train, hidden = c(5, 3),
                 linear.output = TRUE, stepmax = 1e6)
  as.numeric(compute(m, nn_test)$net.result) * (y_max - y_min) + y_min
})
nn_run_mae <- sapply(nn_runs, function(p) mean(abs(test$total_sales - p)))
cat("\nNeural network June MAE across 10 runs:", round(nn_run_mae), "\n")
cat("Range:", round(min(nn_run_mae)), "to", round(max(nn_run_mae)), "\n")

# ----- Compare all models on June -----
results <- rbind(
  score("Baseline (same weekday last week)",   test$total_sales, pred_naive),
  score("Linear regression (trend + weekday)", test$total_sales, pred_lm),
  score("ARIMA (weekly seasonality)",          test$total_sales, pred_arima),
  data.frame(model = "Neural network (avg of 10 runs)",
             MAE  = mean(nn_run_mae),
             RMSE = mean(sapply(nn_runs, function(p) sqrt(mean((test$total_sales - p)^2)))),
             MAPE = mean(sapply(nn_runs, function(p) mean(abs((test$total_sales - p) / test$total_sales)) * 100)))
) %>% arrange(MAE)

cat("\nJune 2023 forecast accuracy (trained on Jan-May):\n")
print(results, row.names = FALSE)

# Chart: June actual vs each model's forecast
june_plot_df <- data.frame(
  date = rep(test$date, 4),
  sales = c(test$total_sales, pred_naive, pred_lm, pred_arima),
  series = rep(c("Actual", "Baseline", "Linear regression", "ARIMA"), each = h)
)
june_plot_df$series <- factor(june_plot_df$series,
                              levels = c("Actual", "Baseline", "Linear regression", "ARIMA"))

p_june <- ggplot(june_plot_df, aes(x = date, y = sales, color = series, linewidth = series)) +
  geom_line() +
  scale_color_manual(values = c("Actual" = "black", "Baseline" = "grey60",
                                "Linear regression" = "#1b9e77", "ARIMA" = "#d95f02")) +
  scale_linewidth_manual(values = c("Actual" = 1.2, "Baseline" = 0.7, "Linear regression" = 0.7,
                                    "ARIMA" = 0.7)) +
  ggtitle("June 2023: Actual vs Forecast (models trained on Jan-May)") +
  xlab("Date") + ylab("Daily Sales ($)") + labs(color = NULL, linewidth = NULL) +
  theme_minimal()
print(p_june)

# ----- Final 30-day forecast: winning model (linear regression) refit on all data -----
lm_full <- lm(total_sales ~ t + day_of_week, data = daily_sales)
future <- data.frame(date = max(daily_sales$date) + 1:30,
                     t    = max(daily_sales$t) + 1:30)
future$day_of_week <- factor(weekdays(future$date), levels = levels(daily_sales$day_of_week))
pi80 <- predict(lm_full, newdata = future, interval = "prediction", level = 0.80)
pi95 <- predict(lm_full, newdata = future, interval = "prediction", level = 0.95)

fc_df <- data.frame(date = future$date, mean = pi95[, "fit"],
                    lo80 = pi80[, "lwr"], hi80 = pi80[, "upr"],
                    lo95 = pi95[, "lwr"], hi95 = pi95[, "upr"])
cat("\nJuly 2023 forecast (linear regression): average", round(mean(fc_df$mean)),
    "per day, from", round(fc_df$mean[1]), "to", round(fc_df$mean[30]), "\n")

p_forecast <- ggplot() +
  geom_ribbon(data = fc_df, aes(x = date, ymin = lo95, ymax = hi95), fill = "steelblue", alpha = 0.2) +
  geom_ribbon(data = fc_df, aes(x = date, ymin = lo80, ymax = hi80), fill = "steelblue", alpha = 0.4) +
  geom_line(data = daily_sales, aes(x = date, y = total_sales), color = "black") +
  geom_line(data = fc_df, aes(x = date, y = mean), color = "blue", linewidth = 1) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b %Y") +
  ggtitle("30-Day Daily Sales Forecast (linear regression: trend + weekday)") +
  xlab("Date") + ylab("Daily Sales ($)") +
  theme_minimal()
print(p_forecast)

# ----- Peak times: day x hour heatmap (units sold) -----
peak_analysis <- coffee.df %>%
  group_by(day_of_week, hour_of_day) %>%
  summarise(total_units = sum(transaction_qty), .groups = "drop") %>%
  arrange(desc(total_units))

cat("\nTop 10 day-hour slots by units sold:\n")
print(head(peak_analysis, 10))

p_heat <- ggplot(peak_analysis, aes(x = hour_of_day, y = day_of_week, fill = total_units)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "#f7fbff", high = "#08306b", name = "Units sold") +
  scale_x_continuous(breaks = 6:20, labels = paste0(6:20, ":00")) +
  scale_y_discrete(limits = rev(levels(coffee.df$day_of_week))) +
  ggtitle("When Customers Buy: Units Sold by Day and Hour (Jan-Jun 2023)") +
  xlab("Hour of day") + ylab(NULL) +
  theme_minimal()
print(p_heat)

# ----- Products and categories -----
top_products <- coffee.df %>%
  group_by(product_detail) %>%
  summarise(total_units   = sum(transaction_qty),
            total_revenue = sum(transaction_qty * unit_price),
            .groups = "drop") %>%
  arrange(desc(total_units))

cat("\nTop 10 products by units sold:\n")
print(head(top_products, 10))

category_performance <- coffee.df %>%
  group_by(product_category) %>%
  summarise(total_units   = sum(transaction_qty),
            total_revenue = sum(transaction_qty * unit_price),
            .groups = "drop") %>%
  mutate(revenue_share = round(100 * total_revenue / sum(total_revenue), 1)) %>%
  arrange(desc(total_revenue))

cat("\nRevenue by product category:\n")
print(category_performance)
