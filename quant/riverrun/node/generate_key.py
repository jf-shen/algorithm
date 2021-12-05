def generate_key(self):
    start_date = self.date_date
    end_date = self.end_date
    count = self.count
    sample_dt = self.sample_dt

    transaction_date = get_price(security=index_code,
                                 start_date=start_date,
                                 end_date=end_date,
                                 count=count).index.tolist()

    sample_date = []
    for i in range(len(transaction_date)):
        if (len(transaction_date) - i - 1) % sample_dt == 0:
            sample_date.append(transaction_date[i])

    df = pd.DataFrame({'code': [], 'date': []})
    for date in sample_date:
        stock_code = get_index_stocks(index_code, date=date)
        df_tmp = pd.DataFrame({'code': stock_code, 'date': date2str(date)})
        df = df.append(df_tmp)

    print('finish key generation')
    return df