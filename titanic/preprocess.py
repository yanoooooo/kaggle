# ./dataにtrain.csvがあること
import pandas as pd

data = pd.read_csv('./data/train.csv')

# nan check
print('** before preprocess **')
print(data.isnull().any(axis=0))
print('data size: %d' % len(data))
print('-'*50)

# delete column
data = data.drop('Name', axis=1)
data = data.drop('Ticket', axis=1)
data = data.drop('Cabin', axis=1)

# replace float
data['Sex'] = data['Sex'].replace({'male': 0.0, 'female': 1.0})
data['Embarked'] = data['Embarked'].replace({'C': 0.0, 'Q': 1.0, 'S': 2.0})

# fill nan
median = data['Age'].median()
data['Age'] = data['Age'].fillna(median)

# delete null row
data = data.dropna()

print('** after preprocess **')
print(data.isnull().any(axis=0))
print('data size: %d' % len(data))
print(data.head())

data.to_csv('./data/treated_train.csv', index=False)
