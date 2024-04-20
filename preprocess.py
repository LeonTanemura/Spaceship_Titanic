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

# 以下は特徴量の作成
def cabin_label(data):
    data['CabinLabel'] = "U-U"
    data.loc[(data['Cabin'].str.match('^A.*P$')), 'CabinLabel'] = "A-P"
    data.loc[(data['Cabin'].str.match('^A.*S$')), 'CabinLabel'] = "A-S"
    data.loc[(data['Cabin'].str.match('^B.*P$')), 'CabinLabel'] = "B-P"
    data.loc[(data['Cabin'].str.match('^B.*S$')), 'CabinLabel'] = "B-S"
    data.loc[(data['Cabin'].str.match('^C.*P$')), 'CabinLabel'] = "C-P"
    data.loc[(data['Cabin'].str.match('^C.*S$')), 'CabinLabel'] = "C-S"
    data.loc[(data['Cabin'].str.match('^D.*P$')), 'CabinLabel'] = "D-P"
    data.loc[(data['Cabin'].str.match('^D.*S$')), 'CabinLabel'] = "D-S"
    data.loc[(data['Cabin'].str.match('^E.*P$')), 'CabinLabel'] = "E-P"
    data.loc[(data['Cabin'].str.match('^E.*S$')), 'CabinLabel'] = "E-S"
    data.loc[(data['Cabin'].str.match('^F.*P$')), 'CabinLabel'] = "F-P"
    data.loc[(data['Cabin'].str.match('^F.*S$')), 'CabinLabel'] = "F-S"
    data.loc[(data['Cabin'].str.match('^G.*P$')), 'CabinLabel'] = "G-P"
    data.loc[(data['Cabin'].str.match('^G.*S$')), 'CabinLabel'] = "G-S"

    data['CabinNum'] = data['Cabin'].str.split("/").str[1]
    data['CabinNum'] = data['CabinNum'].fillna("9999")  # NaNを"9999"で埋める
    data['CabinNum'] = data['CabinNum'].astype(float)

    data = data.drop(['Cabin'], axis=1)
    return data

def passenger_family(data):
    data['FamilyLabel'] = '0' 
    passenger_ids = data["PassengerId"]
    pre_prefix = None
    for idx, passid in enumerate(passenger_ids):
        prefix, suffix = passid.split('_')
        if pre_prefix is not None and prefix == pre_prefix:
            data.loc[data['PassengerId']== passid, 'FamilyLabel'] = '1'
            if suffix == '02':
                data.loc[data['PassengerId'] == passenger_ids.iloc[idx-1], 'FamilyLabel'] = '1'
        pre_prefix = prefix

    return data

# --------------------------------------------------------------------------------
df = cabin_label(df)
df = passenger_family(df)

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
