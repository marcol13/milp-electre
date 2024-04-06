import pandas as pd

def print_scoretable(score_table):
    df = pd.DataFrame(score_table)
    df.rename(columns=lambda x: x.short_name, index=lambda x: x.short_name, inplace=True)
    print(df)