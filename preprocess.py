import pandas as pd
import numpy as np
import statistics as st
import random


train = pd.read_csv("datasets/train.csv")
test = pd.read_csv("datasets/test.csv")

train_test = pd.concat([train, test])


def missing_value_checker(df, name):
    chk_null = df.isnull().sum()
    chk_null_pct = chk_null / (df.index.max() + 1)
    chk_null_tbl = pd.concat([chk_null[chk_null > 0], chk_null_pct[chk_null_pct > 0]], axis=1)
    chk_null_tbl = chk_null_tbl.rename(columns={0: "欠損数",1: "欠損割合"})
    print(name)
    print(chk_null_tbl, end="\n\n")

 #  def homeplanet_isnull(self):
    #     data = self.train
    #     missing_data = data[data.isnull().any(axis=1)]

    #     missing_data.to_csv(to_absolute_path('datasets/train_isnull.csv'), index=False)
    #     print(missing_data)

    #     exit()

def homeplanet_missing_value(data):

    selected_columns = data[['PassengerId','HomePlanet','VIP']]
    pre_prefix = None
    pre_suffix = None
    for idx, (passengerid, homeplanet, vip) in selected_columns.iterrows():
        pre_idx = idx - 1
        prefix, suffix = passengerid.split('_')
        pre_homeplanet = data.iloc[pre_idx, 1]
        if pd.isnull(homeplanet):
            if prefix == pre_prefix and pre_prefix is not None:
                data.loc[data['PassengerId']== passengerid, 'HomePlanet'] = pre_homeplanet
            else:
                next_idx = idx + 1
                if next_idx < len(data):
                    next_passengerid = data.iloc[next_idx, 0]
                    next_homeplanet = data.iloc[next_idx, 1]
                    next_prefix, next_suffix = next_passengerid.split('_') 
                    if next_suffix != '01':
                        if pd.isnull(next_homeplanet):
                            data.loc[data['PassengerId'] == passengerid, 'HomePlanet'] = random.choice(['Earth', 'Europa', 'Mars'])
                        else:
                            data.loc[data['PassengerId'] == passengerid, 'HomePlanet'] = next_homeplanet
                    elif vip:
                        data.loc[data['PassengerId'] == passengerid, 'HomePlanet'] = random.choice(['Europa', 'Mars'])
                    else:
                        data.loc[data['PassengerId'] == passengerid, 'HomePlanet'] = random.choice(['Earth'])     
        pre_prefix = prefix


print(train['HomePlanet'].value_counts())

missing_value_checker(train_test, "train_test")
missing_value_checker(train, "train")
missing_value_checker(test, "test")

# 欠損値の補完(train)
df = train_test
print(df.info())

# 補完するターゲットの設定
targets = []
# mode(最頻値), mean(平均値), median(中央値)
val_name = "mode"

for target in targets:
    if val_name == "mode":
        value = st.mode(df[target])
    elif val_name == "mean":
        value = st.mean(df[target])
    elif val_name == "median":
        value = st.median(df[target])
    else:
        raise ValueError("Invalid value name. Please specify 'mode', 'mean', or 'median'.")
    
    # 欠損値を補完
    train_test[target] = train_test[target].fillna(value)

# 欠損値特徴量の削除
targets = ['Name']
df = df.drop(targets, axis=1)

homeplanet_missing_value(df)
# 変数の型ごとに欠損値の扱いが異なるため、変数ごとに処理
for column in df.columns:
    if df[column].dtype=='O':
        df[column] = df[column].fillna('Unknown')
    elif df[column].dtype=='int64':
        df[column] = df[column].fillna(0)
    elif df[column].dtype=='float64':
        df[column] = df[column].fillna(0.0)
    else:
        raise ValueError("Unsupported dtype encountered. Program terminated.")

train_test = df

# trainとtestに再分割
train = train_test.iloc[:len(train)]
test = train_test.iloc[len(train):]
test = test.drop('Transported', axis=1)

print(train.info())
print(test.info())

# csvファイルの作成
train.to_csv('datasets/train_fix.csv', index=False)
test.to_csv('datasets/test_fix.csv', index=False)
