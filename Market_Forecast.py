import warnings
warnings.filterwarnings('ignore')
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm


stock_ticker = ['AAPL','GOOG','T','MGM','IBM','^GSPC','TSLA','F','AMD','DIS','NEM','MCD']

sp500 = ['^GSPC']

start_date = '2020-10-10'
end_date = '2023-11-21'
risk_free_rate = 0.05

tickers = [sp500 + stock_ticker]

df = yf.download(tickers=stock_ticker, start=start_date, end=end_date)['Adj Close']

def basic_subplots(df):
    plt_, axs = plt.subplots(2,2,figsize=(10,6))
    sns.scatterplot(df,x='GOOG',y='^GSPC',ax=axs[0,0])
    sns.scatterplot(df,x='IBM',y='^GSPC',ax=axs[0,1])
    sns.scatterplot(df,x='T',y='^GSPC',ax=axs[1,0])
    sns.scatterplot(df,x='TSLA',y='^GSPC',ax=axs[1,1])
    plt.show()

basic_subplots(df)

def run_single_index_model(stock_ticker, sp500, risk_free_rate, market_return_forecast):
    stock_data = df[stock_ticker].to_frame()
    sp500_data = df[sp500].to_frame()
    stock_data['Excess_Return'] = stock_data[stock_ticker] - risk_free_rate
    sp500_data['Excess_Return'] = sp500_data['^GSPC'] - risk_free_rate
    model = sm.OLS(endog=stock_data['Excess_Return'], exog=sm.add_constant(sp500_data['Excess_Return'])).fit()
    print(f'Single index for each stock ticker {stock_ticker}:')
    print(model.summary())

    # Calculate adjusted beta
    beta = model.params[1]
    std_resid = model.resid.std()
    adjusted_beta = beta * (1 + (1 + (std_resid**2) / (model.df_resid - 2)) * (model.df_resid / model.df_model)) ** 0.5
    print(f'Adjusted Beta for {stock_ticker}: {adjusted_beta}')

    # Beta Forecast
    forecast_beta = beta * (1 + (std_resid**2 / model.df_resid)) ** 0.5
    print(f'Forecast Beta for {stock_ticker}: {forecast_beta}')

    # RP forecast
    risk_premium_forecast = adjusted_beta * (market_return_forecast - risk_free_rate)
    print(f'Risk Premium Forecast for {stock_ticker}: {risk_premium_forecast}')

    # optimal risky portfolio weight
    optimal_portfolio_weight = adjusted_beta
    print(f'Optimal Risky Portfolio Weight for {stock_ticker}: {optimal_portfolio_weight}')

    # information ratio (not correct)
    information_ratio = (model.params[1] / model.resid.std())
    print(f'Information Ratio for {stock_ticker}: {information_ratio}')

    # variance for placeholder per share(not correct)
    variance_of_eps = 0.05

    # Placeholder for dividend yield
    dividend_yield = 0.02

    # Not accurate either
    debt_to_asset_ratio = 0.4

    print(f'Variance of Earnings per Share for {stock_ticker}: {variance_of_eps}')
    print(f'Dividend Yield for {stock_ticker}: {dividend_yield}')
    print(f'Debt-to-Asset Ratio for {stock_ticker}: {debt_to_asset_ratio}')

    plt.figure(figsize=(10,6))
    sns.scatterplot(x=sp500_data['Excess_Return'], y=stock_data['Excess_Return'], label=stock_ticker)
    sns.lineplot(x=sp500_data['Excess_Return'], y=model.fittedvalues, color='red', label='Security Market Line')
    plt.title(f'Single Index Model for {stock_ticker}')
    plt.xlabel('Market Excess Return')
    plt.ylabel(f'{stock_ticker} Excess Return')
    plt.legend()
    plt.show()

    return {
        'Adjusted Beta': adjusted_beta,
        'Forecast Beta': forecast_beta,
        'Risk Premium Forecast': risk_premium_forecast,
        'Optimal Portfolio Weight': optimal_portfolio_weight,
        'Information Ratio': information_ratio,
        'Variance of Earnings per Share': variance_of_eps,
        'Dividend Yield': dividend_yield,
        'Debt-to-Asset Ratio': debt_to_asset_ratio,
        'Model Residuals': model.resid
    }

# Example usage:
market_return_forecast = 0.08
aapl_results = run_single_index_model('AAPL', '^GSPC', risk_free_rate, market_return_forecast)
tsla_results = run_single_index_model('TSLA', '^GSPC', risk_free_rate, market_return_forecast)
mcd_results = run_single_index_model('MCD', '^GSPC', risk_free_rate, market_return_forecast)

# all results combined
results_combined = {
    'AAPL': aapl_results,
    'TSLA': tsla_results,
    'MCD': mcd_results
}

# Summary of opf
portfolio_weights = {stock: results['Optimal Portfolio Weight'] for stock, results in results_combined.items()}
total_portfolio_weight = sum(portfolio_weights.values())
normalized_portfolio_weights = {stock: weight / total_portfolio_weight for stock, weight in portfolio_weights.items()}
print('\nSummary of Optimization Portfolio:')
