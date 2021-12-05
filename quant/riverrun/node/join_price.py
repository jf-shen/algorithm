from jqdatasdk import get_price
from utils.date_func import date2str, str2date, dplus


def build_node_fn(params, features):

    dt_list = params.get('dt_list')
    field = params.get('field')
    verbose = params.get('verbose', False)
    output_format = params.get('output_format', '%s')

    def node_fn(context, input_list, result_map):
        df = input_list[0]
        print('joining: %s, dt_list = %s' % (field, dt_list))

        date_list = df['date'].unique()
        start_date = dplus(min(date_list), min(dt_list))
        end_date = dplus(max(date_list), max(dt_list))
        stock_code = df['code'].unique().tolist()
        price_df = get_price(stock_code,
                             start_date=str2date(start_date),
                             end_date=str2date(end_date),
                             frequency=frequency,
                             fields=[field],
                             skip_paused=False,
                             fq='pre',
                             count=None,
                             panel=False)

        price_df['T'] = price_df['time'].map(date2str)
        for dt in dt_list:
            if verbose:
                print('append date: %i' % dt)
            price_df['T%i' % dt] = price_df['T'].map(lambda s: dplus(s, -dt))

        for dt in dt_list:
            if verbose:
                print("join %s: date = %s" % (field, dt))
            price = price_df[['T%i' % dt, 'code', field]]
            price.columns = ['date', 'code', output_format % dt]
            df = df.merge(price, how='left', on=['date', 'code'])

    return node_fn


