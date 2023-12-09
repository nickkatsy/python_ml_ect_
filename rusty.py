import yfinance as yf
from pypfopt import expected_returns,EfficientFrontier,risk_models
import warnings
warnings.filterwarnings('ignore')
import riskfolio as rf

stock_tickers = ['AAPL','GOOGL','F','T','TSLA','MSFT','AMZN','NVDA','NFLX']
bond_tickers = ['BND','HYG','TIP','IEF','LQD','SHY','AGG','MUB','HYG','TLT']


start_date = '2013-10-10'
end_date = '2023-12-09'


stocks_df = yf.download(tickers=stock_tickers,start=start_date,end=end_date)['Close']
bonds_df = yf.download(tickers=bond_tickers,start=start_date,end=end_date)['Close']

stock_returns = stocks_df.pct_change().dropna()
bond_returns = bonds_df.pct_change().dropna()

mu_s = expected_returns.mean_historical_return(stocks_df)
cov_s = risk_models.sample_cov(stock_returns)

mu_b = expected_returns.mean_historical_return(bonds_df)
cov_b = risk_models.sample_cov(bonds_df)

ef_s = EfficientFrontier(mu_s,cov_s)
ef_b = EfficientFrontier(mu_b, cov_b)
weights = ef_s.max_sharpe()
cleaned_weights = ef_s.clean_weights()
print(cleaned_weights)
print(cov_b)
print(weights)
print(mu_b)
print(mu_s)
print(cov_b)


# risky stuffff

risky_assets = stocks_df.copy()
risk_free_assets = bonds_df.copy()

port_risky = rf.Portfolio(returns=risky_assets)
port_risk_free = rf.Portfolio(returns=risk_free_assets)

port_risky.assets_stats(method_mu='hist',method_cov='hist',d=0.94)
w = port_risky.optimization(model='Classic',rm='MV',obj='Sharpe',rf=0.05,l=2,hist=True)

port_risk_free.assets_stats(method_mu='hist',method_cov='hist',d=0.94)
w1 = port_risk_free.optimization(model='Classic',rm='MV',obj='Sharpe',rf=0.05,l=2,hist=True)







