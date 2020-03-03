import pandas as pd
data = {'col_1': 0, 'col_2': None}
for key in data.keys():
    data[key] = list(data[key])
pd.DataFrame.from_dict(data)