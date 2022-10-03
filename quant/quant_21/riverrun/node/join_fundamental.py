def join_fundamental(df, verbose=True):
    date_list = df['date'].unique()
    res_df = None
    for ds in date_list:
        if verbose:
            print('joining fundamental: date = %s' % ds)
        sample = get_index_stocks(index_code, date=str2date(ds))
        q = query(
            valuation.code,
            valuation.market_cap,
            balance.total_assets,
            balance.total_liability,
            indicator.inc_revenue_year_on_year,
            balance.development_expenditure
        ).filter(
            valuation.code.in_(sample)
        )

        fundamental_df = get_fundamentals(q, date=str2date(ds))
        fundamental_df['date'] = ds
        if res_df is None:
            res_df = fundamental_df
        else:
            res_df = res_df.append(fundamental_df)

    df = df.merge(res_df, how='left', on=['date', 'code'])
    return df
