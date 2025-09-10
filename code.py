import pandas as pd
import statsmodels.api as sm

# Example: df has columns ["date", "spy", "AAPL", "MSFT", "GOOG", ...]
# Set index to date if not already
df = df.set_index("date")

# Compute returns (log or simple, here simple % change)
returns = df.pct_change().dropna()

# Function to compute rolling beta of one stock vs SPY
def rolling_beta(stock_returns, market_returns, window=52):
    betas = []
    for i in range(window, len(stock_returns)+1):
        y = stock_returns.iloc[i-window:i]
        x = market_returns.iloc[i-window:i]
        x = sm.add_constant(x)  # add intercept
        model = sm.OLS(y, x).fit()
        betas.append(model.params[1])  # slope = beta
    # Align result to original index
    beta_series = pd.Series([None]*(window-1) + betas, index=stock_returns.index)
    return beta_series

# Apply to each stock (excluding "spy")
betas_df = pd.DataFrame(index=returns.index)

for stock in returns.columns:
    if stock != "spy":
        betas_df[stock] = rolling_beta(returns[stock], returns["spy"], window=52)


import pandas as pd

# Example: df has columns [date, stockname, price]
# Ensure it's sorted
df = df.sort_values(["stockname", "date"]).copy()

# Returns
df["return"] = df.groupby("stockname")["price"].pct_change()

# Extract market (SP500) returns
mkt = (
    df.loc[df["stockname"] == "SP500", ["date", "return"]]
    .rename(columns={"return": "mkt_return"})
)

# Merge market returns back into df
df = df.merge(mkt, on="date", how="left")

# Rolling beta using cov/var
window = 30
df["cov"] = df.groupby("stockname")["return"].rolling(window).cov(df["mkt_return"]).reset_index(level=0, drop=True)
df["var_mkt"] = df["mkt_return"].rolling(window).var()
df["beta"] = df["cov"] / df["var_mkt"]

# Optional: drop beta for SP500 itself
df.loc[df["stockname"] == "SP500", "beta"] = None



window = 30

# Compute covariance grouped by stockname
cov = df.groupby("stockname")["return"].rolling(window).cov(df["mkt_return"])

# Remove the group index and align with df
df["cov"] = cov.reset_index(level=0, drop=True)

# Market variance (same for all stocks, so no groupby needed)
df["var_mkt"] = df["mkt_return"].rolling(window).var()

# Beta
df["beta"] = df["cov"] / df["var_mkt"]

# Drop beta for SP500 itself
df.loc[df["stockname"] == "SP500", "beta"] = None
import pandas as pd

# Ensure proper datatypes
df['date'] = pd.to_datetime(df['date'])
df['time'] = pd.to_datetime(df['time']).dt.time

# Helper to get last close of previous day
df = df.sort_values(['sym', 'date', 'time'])
df['prev_close'] = df.groupby('sym')['price'].shift(1)
df['prev_date'] = df.groupby('sym')['date'].shift(1)

# Only keep prev_close from the last row of previous day
df['prev_close'] = df.groupby(['sym', 'date'])['prev_close'].transform('first')

# Pivot to make it easier to reference times
pivoted = df.pivot_table(index=['date', 'sym'], 
                         columns='time', 
                         values='price')

# Extract relevant times
o   = pivoted[ pd.to_datetime("09:30").time() ]
o30 = pivoted[ pd.to_datetime("10:00").time() ]
c60 = pivoted[ pd.to_datetime("15:00").time() ]
c30 = pivoted[ pd.to_datetime("15:30").time() ]
c   = pivoted[ pd.to_datetime("16:00").time() ]
pc  = df.groupby(['date','sym'])['prev_close'].first()

# Compute returns
res = pd.DataFrame(index=pivoted.index)
res['ON']    = o / pc - 1
res['FH']    = o30 / o - 1
res['M']     = c60 / o30 - 1
res['SLH']   = c30 / c60 - 1
res['LH']    = c / c30 - 1
res['ONFH']  = o30 / pc - 1
res['ROD3']  = c60 / pc - 1

# Merge back with your daily data (returns, beta etc.)
final = df.drop_duplicates(['date','sym']) \
          .set_index(['date','sym']) \
          .join(res)



# Pivot so each time is a column
prices_wide = intraday.pivot_table(
    index=["sym", "date"], 
    columns="time", 
    values="price"
).reset_index()

# Make sure columns are easy to access
prices_wide.columns.name = None


prices_wide["prev_1630"] = prices_wide.groupby("sym")["16:30"].shift(1)
import numpy as np

# ROD3: prev 16:30 → today 15:00
prices_wide["ROD3"] = np.log(prices_wide["15:00"] / prices_wide["prev_1630"])

# ROD4: prev 16:30 → today 15:30
prices_wide["ROD4"] = np.log(prices_wide["15:30"] / prices_wide["prev_1630"])

# SLH: 15:00 → 15:30
prices_wide["SLH"] = np.log(prices_wide["15:30"] / prices_wide["15:00"])

# LH: 15:30 → 16:00
prices_wide["LH"] = np.log(prices_wide["16:00"] / prices_wide["15:30"])
returns = prices_wide[["sym", "date", "ROD3", "ROD4", "SLH", "LH"]]


# Pivot intraday
prices_wide = intraday.pivot_table(
    index=["sym", "date"], 
    columns="time", 
    values="price"
).reset_index()

# Flatten column names
prices_wide.columns.name = None

# Inspect column names (likely '16:30:00')
print(prices_wide.columns)

# Use the exact string that appears in your columns
col_1630 = "16:30:00"
col_1500 = "15:00:00"
col_1530 = "15:30:00"
col_1600 = "16:00:00"

# Shift previous day's 16:30 per symbol
prices_wide["prev_1630"] = prices_wide.groupby("sym")[col_1630].shift(1)

# Compute returns
import numpy as np

prices_wide["ROD3"] = np.log(prices_wide[col_1500] / prices_wide["prev_1630"])
prices_wide["ROD4"] = np.log(prices_wide[col_1530] / prices_wide["prev_1630"])
prices_wide["SLH"]  = np.log(prices_wide[col_1530] / prices_wide[col_1500])
prices_wide["LH"]   = np.log(prices_wide[col_1600] / prices_wide[col_1530])

# Final dataset
returns = prices_wide[["sym", "date", "ROD3", "ROD4", "SLH", "LH"]]
import pandas as pd
import numpy as np

# make sure datetime is sorted
df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
df = df.sort_values(["sym", "date", "datetime"])

# compute log returns within each sym+date
df["log_ret"] = df.groupby(["sym", "date"])["price"].apply(lambda x: np.log(x / x.shift(1)))

# drop the first NA of each day
df = df.dropna(subset=["log_ret"])

# realized volatility = sum of squared log returns per sym+date
rv = (
    df.groupby(["sym", "date"])["log_ret"]
      .apply(lambda x: (x**2).sum())
      .reset_index(name="realized_vol")
)

print(rv.head())

