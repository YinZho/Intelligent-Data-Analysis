import pandas as pd

df = pd.read_csv(
    filepath_or_buffer='e:/2-summer/iris.data.csv')

df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']

df.tail()