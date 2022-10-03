
# label_dt_list = [1, 3, 7, 14, 28, 90]
label_dt_list = [1, 5]
feature_dt_list = [0, -1, -2, -3, -4, -5, -6, -7]

PIPELINE = {
  "sample_builder": [
    {
      "type": "join_price",
      "dt_list": label_dt_list,
      "output_format": "_price_%id",
      "field": "open",
      "skip_online": True,
      "verbose": False
    },
    {
      "type": "join_price",
      "dt_list": feature_dt_list,
      "verbose": False,
      "field": "open",
      "output_format": "price_%id"
    },
    {
      "type": "join_index",
      "dt_list": label_dt_list,
      "output_format": "_index_%id",
      "field": "open",
      "skip_online": True,
      "verbose": False
    },
    {
      "type": "join_index",
      "dt_list": feature_dt_list,
      "verbose": False,
      "field": "open",
      "output_format": "index_%id"
    },
    {
      "type": "join_price",
      "dt_list": feature_dt_list,
      "verbose": False,
      "field": "volume",
      "output_format": "volume_%id"
    },
    {
      "type": "join_fundamental",
      "verbose": False
    },
    {
      "type": "join_industry"
    }
  ]
}
