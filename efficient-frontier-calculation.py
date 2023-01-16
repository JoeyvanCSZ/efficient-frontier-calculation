import pandas as pd
import yfinance as yf
import pypfopt
import matplotlib.pyplot as plt
import math
import numpy as np
import pprint
import sys

from pypfopt import objective_functions
from datetime import datetime, date, timedelta
from pypfopt import plotting
from pypfopt import EfficientFrontier
from pypfopt import expected_returns
from pypfopt import risk_models
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import EfficientSemivariance

class Program:
	def __init__(self, tickers, window_size, total_portfolio_value, result_file):
		self.tickers = tickers
		self.window_size = window_size
		self.total_portfolio_value = total_portfolio_value
		self.result_file = open(result_file, 'w')

		self.num_trading_days_per_year = 252
		self.window_size_adjudged = math.ceil(self.window_size * (365 / self.num_trading_days_per_year)) + 1
		self.date_format = "%Y-%m-%d"
		self.end_date = datetime.today().strftime(self.date_format)
		self.start_date = (datetime.today() - timedelta(days=self.window_size_adjudged)).strftime(self.date_format)

		print("Tickers: {}".format(tickers), file=self.result_file)
		print("Window size: {}".format(window_size), file=self.result_file)
		print("Total portfolio value: {}".format(total_portfolio_value), file=self.result_file)

		self.fecth_price_data_and_preproccess();

	def __del__(self):
		self.result_file.close()

	def fecth_price_data_and_preproccess(self):
		ohlc = yf.download(self.tickers, self.start_date, self.end_date, progress=False)
		days = "{}D".format(self.window_size);
		prices = ohlc["Adj Close"].dropna(how="all")
		prices = prices.first(days)

		self.mu = expected_returns.mean_historical_return(prices, frequency=self.num_trading_days_per_year)
		self.S = risk_models.sample_cov(prices, frequency=self.num_trading_days_per_year)
		self.historical_returns = expected_returns.returns_from_prices(prices)
		self.latest_prices = get_latest_prices(prices)

	def max_sharpe(self):
		print("\n========== Maximises Sharpe ==========", file=self.result_file)
		try:
			ef = EfficientFrontier(self.mu, self.S)
			ef.max_sharpe()
			clean_weights = ef.clean_weights()
			performance = ef.portfolio_performance()
			da = DiscreteAllocation(clean_weights, self.latest_prices, total_portfolio_value=self.total_portfolio_value)
			allocation, leftover = da.greedy_portfolio()

			removeKeys = []
			for key, value in clean_weights.items():
				if(value == 0):
					removeKeys.append(key)
			for key in removeKeys:
				del clean_weights[key]

			print("Expected annual return: {:.2%}".format(performance[0]), file=self.result_file)
			print("Annual volatility: {:.2%}".format(performance[1]), file=self.result_file)
			print("Sharpe Ratio: {:.2f}".format(performance[2]), file=self.result_file)
			print("Weights: {}".format([(k,"{:.2%}".format(v)) for k,v in clean_weights.items()]), file=self.result_file)
			print("Discrete allocation:", allocation, file=self.result_file)
			print("Funds remaining: ${:.2f}".format(leftover), file=self.result_file)
			return True, performance[0], performance[1], performance[2];
		except pypfopt.exceptions.OptimizationError as e:
			print("Optimization failed.", file=self.result_file)
			return False, 0, 0, 0;

	def min_volatility(self):
		print("\n========== Minimise Volatility ==========", file=self.result_file)
		try:
			ef = EfficientFrontier(self.mu, self.S)
			ef.min_volatility()
			clean_weights = ef.clean_weights()
			performance = ef.portfolio_performance()
			da = DiscreteAllocation(clean_weights, self.latest_prices, total_portfolio_value=self.total_portfolio_value)
			allocation, leftover = da.greedy_portfolio()

			removeKeys = []
			for key, value in clean_weights.items():
				if(value == 0):
					removeKeys.append(key)
			for key in removeKeys:
				del clean_weights[key]

			
			print("Expected annual return: {:.2%}".format(performance[0]), file=self.result_file)
			print("Annual volatility: {:.2%}".format(performance[1]), file=self.result_file)
			print("Sharpe Ratio: {:.2f}".format(performance[2]), file=self.result_file)
			print("Weights: {}".format([(k,"{:.2%}".format(v)) for k,v in clean_weights.items()]), file=self.result_file)
			print("Discrete allocation:", allocation, file=self.result_file)
			print("Funds remaining: ${:.2f}".format(leftover), file=self.result_file)
			return True, performance[0], performance[1], performance[2];
		except pypfopt.exceptions.OptimizationError as e:
			print("Optimization failed.", file=self.result_file)
			return False, 0, 0, 0;

	def min_semivariance(self):
		print("\n========== Minimise Semivariance ==========", file=self.result_file)
		try:
			es = EfficientSemivariance(self.mu, self.historical_returns)
			es.min_semivariance()
			clean_weights = es.clean_weights()
			performance = es.portfolio_performance()
			da = DiscreteAllocation(clean_weights, self.latest_prices, total_portfolio_value=self.total_portfolio_value)
			allocation, leftover = da.greedy_portfolio()

			removeKeys = []
			for key, value in clean_weights.items():
				if(value == 0):
					removeKeys.append(key)
			for key in removeKeys:
				del clean_weights[key]

			print("Expected annual return: {:.2%}".format(performance[0]), file=self.result_file)
			print("Annual volatility: {:.2%}".format(performance[1]), file=self.result_file)
			print("Sharpe Ratio: {:.2f}".format(performance[2]), file=self.result_file)
			print("Weights: {}".format([(k,"{:.2%}".format(v)) for k,v in clean_weights.items()]), file=self.result_file)
			print("Discrete allocation:", allocation, file=self.result_file)
			print("Funds remaining: ${:.2f}".format(leftover), file=self.result_file)
			return True, performance[0], performance[1], performance[2];
		except pypfopt.exceptions.OptimizationError as e:
			print("Optimization failed.", file=self.result_file)
			return False, 0, 0, 0;

	def semivariance_efficient_return(self, target_annul_return=0.1):
		print("\n========== Semivariance Efficient Return ==========", file=self.result_file)
		try:
			target_annul_return = abs(target_annul_return)

			es = EfficientSemivariance(self.mu, self.historical_returns)
			es.efficient_return(target_annul_return)
			clean_weights = es.clean_weights()
			performance = es.portfolio_performance()
			da = DiscreteAllocation(clean_weights, self.latest_prices, total_portfolio_value=self.total_portfolio_value)
			allocation, leftover = da.greedy_portfolio()

			removeKeys = []
			for key, value in clean_weights.items():
				if(value == 0):
					removeKeys.append(key)
			for key in removeKeys:
				del clean_weights[key]

			print("Expected annual return: {:.2%}".format(performance[0]), file=self.result_file)
			print("Annual volatility: {:.2%}".format(performance[1]), file=self.result_file)
			print("Sharpe Ratio: {:.2f}".format(performance[2]), file=self.result_file)
			print("Weights: {}".format([(k,"{:.2%}".format(v)) for k,v in clean_weights.items()]), file=self.result_file)
			print("Discrete allocation:", allocation, file=self.result_file)
			print("Funds remaining: ${:.2f}".format(leftover), file=self.result_file)
			return True, performance[0], performance[1], performance[2];
		except pypfopt.exceptions.OptimizationError as e:
			print("Optimization failed.", file=self.result_file)
			return False, 0, 0, 0;

	def semivariance_efficient_risk(self, target_semideviation=0.1):
		print("\n========== Semivariance Efficient Risk ==========", file=self.result_file)
		try:
			es = EfficientSemivariance(self.mu, self.historical_returns)
			es.efficient_risk(target_semideviation)
			clean_weights = es.clean_weights()
			performance = es.portfolio_performance()
			da = DiscreteAllocation(clean_weights, self.latest_prices, total_portfolio_value=self.total_portfolio_value)
			allocation, leftover = da.greedy_portfolio()

			removeKeys = []
			for key, value in clean_weights.items():
				if(value == 0):
					removeKeys.append(key)
			for key in removeKeys:
				del clean_weights[key]

			print("Expected annual return: {:.2%}".format(performance[0]), file=self.result_file)
			print("Annual volatility: {:.2%}".format(performance[1]), file=self.result_file)
			print("Sharpe Ratio: {:.2f}".format(performance[2]), file=self.result_file)
			print("Weights: {}".format([(k,"{:.2%}".format(v)) for k,v in clean_weights.items()]), file=self.result_file)
			print("Discrete allocation:", allocation, file=self.result_file)
			print("Funds remaining: ${:.2f}".format(leftover), file=self.result_file)
			return True, performance[0], performance[1], performance[2];
		except pypfopt.exceptions.OptimizationError as e:
			print("Optimization failed.", file=self.result_file)
			return False, 0, 0, 0;

if __name__ == '__main__':
	if len(sys.argv) <= 4:
		tickers = ["QLD", "SSO", "DDM", "UBT"]
		window_size = 365
		total_portfolio_value = 1000000
		result_file = "result.txt"
	else:
		tickers = sys.argv[1].split(',')
		for i in range(len(tickers)):
			tickers[i] = tickers[i].strip().upper()
		window_size = int(sys.argv[2])
		total_portfolio_value = int(sys.argv[3])
		result_file = sys.argv[4]

	program = Program(tickers, window_size, total_portfolio_value, result_file)

	# max sharpe
	(max_sharpe_success, max_sharpe_annual_return, max_sharpe_volatility, max_sharpe_sharpe_ratio) = program.max_sharpe()

	# min volatility
	(min_volatility_success, min_volatility_annual_return, min_volatility_volatility, min_volatility_sharpe_ratio) = program.min_volatility()

	# min semivariance
	(min_semivariance_success, min_semivariance_annual_return, min_semivariance_volatility, min_semivariance_sharpe_ratio) = program.min_semivariance()

	# semivariance efficient return
	target_annual_return = 0
	if max_sharpe_success and min_volatility_success:
		total_annual_return = abs(max_sharpe_annual_return) + abs(min_volatility_annual_return)
		target_annual_return += max_sharpe_annual_return * (abs(max_sharpe_annual_return) / total_annual_return)
		target_annual_return += min_volatility_annual_return * (abs(min_volatility_annual_return) / total_annual_return)
	elif max_sharpe_success:
		target_annual_return += max_sharpe_annual_return
	elif min_volatility_success:
		target_annual_return += min_volatility_annual_return

	(efficient_return_success, efficient_return_annual_return, efficient_return_volatility, efficient_return_sharpe_ratio) = program.semivariance_efficient_return(target_annual_return)

	# semivariance efficient risk
	target_semideviation = 0
	if max_sharpe_success and min_semivariance_success:
		total_semideviation = abs(max_sharpe_volatility) + abs(min_semivariance_volatility)
		target_semideviation += max_sharpe_volatility * (abs(max_sharpe_volatility) / total_semideviation)
		target_semideviation += min_semivariance_volatility * (abs(min_semivariance_volatility) / total_semideviation)
	elif max_sharpe_success:
		target_semideviation += max_sharpe_volatility
	elif min_semivariance_success:
		target_semideviation += min_semivariance_volatility

	(efficient_risk_success, efficient_risk_annual_return, efficient_risk_volatility, efficient_risk_sharpe_ratio) = program.semivariance_efficient_risk(target_semideviation)