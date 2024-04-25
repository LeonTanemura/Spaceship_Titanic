import pandas as pd
import numpy as np
import statistics as st
import random

from sklearn.impute import KNNImputer

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
                            if vip:
                                data.loc[data['PassengerId'] == passengerid, 'HomePlanet'] = random.choice(['Europa', 'Mars'])
                            else:
                                data.loc[data['PassengerId'] == passengerid, 'HomePlanet'] = 'Earth'
                        else:
                            data.loc[data['PassengerId'] == passengerid, 'HomePlanet'] = next_homeplanet
                    elif vip:
                        data.loc[data['PassengerId'] == passengerid, 'HomePlanet'] = random.choice(['Europa', 'Mars'])
                    else:
                        data.loc[data['PassengerId'] == passengerid, 'HomePlanet'] = 'Earth'     
        pre_prefix = prefix


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
    data.loc[(data['Cabin'].str.match('^T.*P$')), 'CabinLabel'] = "T-P"
    data.loc[(data['Cabin'].str.match('^T.*S$')), 'CabinLabel'] = "T-S"

    data['CabinNum'] = data['Cabin'].str.split("/").str[1]
    data['CabinNum'] = data['CabinNum'].fillna("9999")  # NaNを"9999"で埋める
    data['CabinNum'] = data['CabinNum'].astype(float)

    data = data.drop(['Cabin'], axis=1)
    return data

def cabin_label2(data):
    data['CabinLabelLeft'] = "U"
    data['CabinLabelRight'] = "U"
    data.loc[(data['Cabin'].str.match('^A.*')), 'CabinLabelLeft'] = "A"
    data.loc[(data['Cabin'].str.match('^B.*')), 'CabinLabelLeft'] = "B"
    data.loc[(data['Cabin'].str.match('^C.*')), 'CabinLabelLeft'] = "C"
    data.loc[(data['Cabin'].str.match('^D.*')), 'CabinLabelLeft'] = "D"
    data.loc[(data['Cabin'].str.match('^E.*')), 'CabinLabelLeft'] = "E"
    data.loc[(data['Cabin'].str.match('^F.*')), 'CabinLabelLeft'] = "F"
    data.loc[(data['Cabin'].str.match('^G.*')), 'CabinLabelLeft'] = "G"
    data.loc[(data['Cabin'].str.match('^T.*')), 'CabinLabelLeft'] = "T"

    data.loc[(data['Cabin'].str.match('.*S$')), 'CabinLabelRight'] = "S"
    data.loc[(data['Cabin'].str.match('.*P$')), 'CabinLabelRight'] = "P"

    data['CabinNum'] = data['Cabin'].str.split("/").str[1]
    data['CabinNum'] = data['CabinNum'].fillna(9999)  # NaNを"9999"で埋める
    data['CabinNum'] = data['CabinNum'].astype(float)
    data['CabinRegion1'] = (data['CabinNum'] < 300).astype(int)
    data['CabinRegion2'] = ((data['CabinNum'] >= 300) & (data['CabinNum'] < 600)).astype(int)
    data['CabinRegion3'] = ((data['CabinNum'] >= 600) & (data['CabinNum'] < 900)).astype(int)
    data['CabinRegion4'] = ((data['CabinNum'] >= 900) & (data['CabinNum'] < 1200)).astype(int)
    data['CabinRegion5'] = ((data['CabinNum'] >= 1200) & (data['CabinNum'] < 1500)).astype(int)
    data['CabinRegion6'] = ((data['CabinNum'] >= 1500) & (data['CabinNum'] < 1800)).astype(int)
    data['CabinRegion7'] = ((data['CabinNum'] >= 1800) & (data['CabinNum'] < 2000)).astype(int)

    data = data.drop(['Cabin'], axis=1)
    data = data.drop(['CabinNum'], axis=1) 
    return data

def cabin_completion(data):
    for _, group in df.groupby('RoomNum'):
        missing_cabins = group[group['Cabin'].isnull()]
        if not missing_cabins.empty:
            non_null_cabins = group.dropna(subset=['Cabin'])
            if not non_null_cabins.empty:
                cabin_to_fill = non_null_cabins.iloc[0]['Cabin']
                data.loc[missing_cabins.index, 'Cabin'] = cabin_to_fill

    data['Cabin'] = data['Cabin'].fillna('Unknown')
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

def surname(data):
    data['Name'] = data['Name'].fillna('Unknown Unknown')
    data['Surname'] = data['Name'].str.split().str[-1]
    data['FamilySize'] = data['Surname'].map(lambda x: data['Surname'].value_counts()[x])
    data.loc[data['Surname'] == 'Unknown','Surname']=np.nan
    data.loc[data['FamilySize'] > 100,'FamilySize']= 0
    data = data.drop(['Name'], axis=1)
    return data

def room_number(data):
    data['RoomNum'] = data['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)
    data['RoomSize'] = data['RoomNum'].map(lambda x: data['RoomNum'].value_counts()[x])
    return data

# --------------------------------------------------------------------------------
# 欠損値割合チェック
missing_value_checker(train_test, "train_test")

# 欠損値の補完
df = train_test

df = room_number(df)
df = cabin_completion(df)
df = cabin_label2(df)
df = passenger_family(df)
df = surname(df)


targets = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df.loc[df['CryoSleep'] == True, targets] = 0
df['MoneyTotal'] = df[targets].sum(axis=1)
df['MoneyLabel'] = np.where(df['MoneyTotal'] != 0, 1, 0)
df.loc[df['MoneyTotal'] != 0, 'CryoSleep'] = False
df.loc[df['HomePlanet'].isnull() & (df['MoneyTotal'] > 10000), 'HomePlanet'] = "Europa"

df.loc[(df['MoneyTotal'] == 0) & (df['FamilyLabel'] == "0"), 'CryoSleep'] = True

targets = ["A", "B", "C", "T"]
for target in targets:
    df.loc[(df['HomePlanet'].isnull()) & (df['CabinLabelLeft'].str.contains(target)), 'HomePlanet'] = "Europa"
targets = ["F"]
for target in targets:
    df.loc[(df['HomePlanet'].isnull()) & (df['CabinLabelLeft'].str.contains(target)) & (df['CryoSleep'] == True), 'HomePlanet'] = "Mars"
targets = ["G"]
for target in targets:
    df.loc[(df['HomePlanet'].isnull()) & (df['CabinLabelLeft'].str.contains(target)), 'HomePlanet'] = "Earth"

df.loc[df['HomePlanet'] == "Earth", 'VIP'] = False

homeplanet_missing_value(df)

missing_value_checker(df, "dataset")
print(df.info())
df['CryoSleep'] = df['CryoSleep'].fillna("True")
pd.set_option('future.no_silent_downcasting', True)
targets = ['HomePlanet', 'VIP', 'Destination']
for target in targets:
    value = df[target].mode().iloc[0]
    df[target] = df[target].fillna(value)

targets = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall',  'VRDeck', 'Spa']
imputer=KNNImputer(n_neighbors=5)
imputer.fit(df[targets])
df[targets]=imputer.transform(df[targets])
# for target in targets:
#     value = np.nanmean(df[target])
#     df[target] = df[target].fillna(value)
# missing_value_checker(df, "dataset")

# targets = ['RoomService', 'FoodCourt', 'ShoppingMall',  'VRDeck', 'Spa', "MoneyTotal"]
# for target in targets:
#     df[target]=np.log(1+df[target])


for column in df.columns:
    if df[column].dtype=='O':
        df[column] = df[column].fillna('Unknown')
    elif df[column].dtype=='int64':
        df[column] = df[column].fillna(0)
    elif df[column].dtype=='float64':
        df[column] = df[column].fillna(0.0)
    else:
        raise ValueError("Unsupported dtype encountered. Program terminated.")

df.to_csv('datasets/concat_fix.csv', index=False)

# targets = ['RoomNum', 'Surname', 'FamilySize', 'RoomSize', 'MoneyTotal']
targets = ['RoomNum', "Name", "FamilyLabel"]
df = df.drop(targets, axis=1)
missing_value_checker(df, "dataset")
print(df.info())
train_test = df

# trainとtestに再分割
train = train_test.iloc[:len(train)]
test = train_test.iloc[len(train):]
test = test.drop('Transported', axis=1)

# csvファイルの作成
train.to_csv('datasets/train_fix2.csv', index=False)
test.to_csv('datasets/test_fix2.csv', index=False)
