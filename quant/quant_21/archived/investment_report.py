import numpy as np 
import datetime


def get_report(buy_date, buy_worth, sell_date, sell_worth):  
	datetime_buy = datetime.datetime.strptime(buy_date, '%Y-%m-%d')
	datetime_sell = datetime.datetime.strptime(sell_date, '%Y-%m-%d')

	days = (datetime_sell - datetime_buy).days
	assert days > 0

	net_return = sell_worth - buy_worth
	return_rate = net_return / buy_worth 
	annualized_return_rate = (1 + return_rate) ** (1 / days * 365) - 1 
	
	print('net_return: ', net_return)
	print('return_rate:  %.1f%%' % (return_rate * 100))  

	print('holding days: ', days)
	print('annualized_return_rate: %.1f%%' % (annualized_return_rate * 100))


if __name__ == "__main__":
	buy_date = '2020-05-25'
	sell_date =  '2021-10-10'

	buy_worth = 1000
	sell_worth = 1140

	get_report(buy_date=buy_date, buy_worth=buy_worth, sell_date=sell_date, sell_worth=sell_worth)


