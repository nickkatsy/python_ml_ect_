import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import yfinance as yf


stock_ticker = ['AAPL','GOOG','T','MGM','IBM','^GSPC','TSLA','F','AMD','DIS','NEM','MCD']

sp500 = ['^GSPC']



start_date = '2020-10-10'

end_date = '2023-11-10'

risk_free_rate = 0.05

tickers = [sp500 + stock_ticker]

df = yf.download(tickers=stock_ticker,start=start_date,end=end_date)['Adj Close']


import seaborn as sns
import matplotlib.pyplot as plt


def basic_subplots(df):
    plt_,axs = plt.subplots(2,2,figsize=(10,6))
    sns.scatterplot(df,x='GOOG',y='^GSPC',ax=axs[0,0])
    sns.scatterplot(df,x='IBM',y='^GSPC',ax=axs[0,1])
    sns.scatterplot(df,x='T',y='^GSPC',ax=axs[1,0])
    sns.scatterplot(df,x='TSLA',y='^GSPC',ax=axs[1,1])
    plt.show()


basic_subplots(df)




import statsmodels.api as sm


def run_single_index_model(stock_ticker, sp500, risk_free_rate):
    stock_data = df[stock_ticker].to_frame()
    sp500_data = df[sp500].to_frame()
    stock_data['Excess_Return'] = stock_data[stock_ticker] - risk_free_rate
    sp500_data['Excess_Return'] = sp500_data['^GSPC'] - risk_free_rate
    model = sm.OLS(endog=stock_data['Excess_Return'], exog=sm.add_constant(sp500_data['Excess_Return'])).fit()
    print(f'single index for each stock ticker{stock_data}')
    print(model.summary())
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=sp500_data['Excess_Return'], y=stock_data['Excess_Return'], label=stock_ticker)
    sns.lineplot(x=sp500_data['Excess_Return'], y=model.fittedvalues, color='red', label='Security Market Line')
    plt.title(f'Single Index Model for {stock_ticker}')
    plt.xlabel('Market Excess Return')
    plt.ylabel(f'{stock_ticker} Excess Return')
    plt.legend()
    plt.show()


#For Apple
run_single_index_model('AAPL','^GSPC', risk_free_rate)

#For Tesla
run_single_index_model('TSLA','^GSPC', risk_free_rate)


#For Mickey Ds
run_single_index_model('MCD','^GSPC', risk_free_rate)


# all stocks together
X = df.drop(['^GSPC'],axis=1)
y = df[['^GSPC']]
model_full = sm.OLS(endog=y,exog=sm.add_constant(X)).fit()

# All together now
print('\nSummary for All Tickers Together:')
print(model_full.summary())









