def join_industry(df):
    # don't take date into consideration for now
    date_list = df['date'].unique()
    industry_list = ['801010', '801020', '801030', '801040', '801050', '801080', '801110', '801120', '801130',
                     '801140', '801150', '801160', '801170', '801180', '801200', '801210', '801230', '801710',
                     '801720', '801730', '801740', '801750', '801760', '801770', '801780', '801790', '801880',
                     '801890']
    industry_df = None
    for industry_code in industry_list:
        stock_code = get_industry_stocks(industry_code, date=str2date(max(date_list)))
        industry_stocks = pd.DataFrame({'code': stock_code, 'industry_code': industry_code})
        if industry_df is None:
            industry_df = industry_stocks
        else:
            industry_df = industry_df.append(industry_stocks)

    assert industry_df.shape[0] == industry_df['code'].unique().shape[0]  # a stock belongs only to one industry
    df = df.merge(industry_df, how='left', on='code')
    return df