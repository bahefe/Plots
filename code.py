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
