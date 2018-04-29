import pandas as pd
import seaborn as sns
import os 
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
 
#zbior treningowy 
print('zbior treningowy')
rtrain = pd.read_csv(os.path.join('train', 'train.tsv'), sep='\t', names=["Occupancy", "date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])
 
print(rtrain.describe())
print('-' * 100)
print('Occupancy % :', end='')
print(sum(rtrain.Occupancy) / len(rtrain))
print('dokładność modelu Zero rule na zestawie treningowym',
       1 - sum(rtrain.Occupancy) / len(rtrain))
print('-' * 100)
print('liczba probek w ktorych osoba sie znajduje :', end='')
print(sum(rtrain.Occupancy))
 
 
lr = LogisticRegression()
lr.fit(rtrain.CO2.values.reshape(-1, 1), rtrain.Occupancy)
 
print('dokoladnosc modelu lr dla jednej zmiennej CO2 na danych treningowych: ', end='')
print(sum(lr.predict(rtrain.CO2.values.reshape(-1, 1))
           == rtrain.Occupancy) / len(rtrain)) 
print('-'*100)
 
lr_full = LogisticRegression()
X = pd.DataFrame(
    rtrain, columns=['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio'])
lr_full.fit(X, rtrain.Occupancy)
print('dokladnosc modelu lr dla wszystkich zmiennych oprocz daty na danych treningowych: ', end='')
print(sum(lr_full.predict(X) == rtrain.Occupancy) / len(rtrain))
print('True Positives: ', end ='')
print(sum((lr_full.predict(X) == rtrain.Occupancy) & (lr_full.predict(X) == 1)))
print('True Negatives: ', end ='')
print(sum((lr_full.predict(X) == rtrain.Occupancy) & (lr_full.predict(X) == 0)))
print('False Positives: ', end ='')
print(sum((lr_full.predict(X) != rtrain.Occupancy) & (lr_full.predict(X) == 1)))
print('False Negatives: ', end ='')
print(sum((lr_full.predict(X) != rtrain.Occupancy) & (lr_full.predict(X) == 0)))
print('-'*100)
print('-'*100)
print('-'*100)
 
#zbior deweloperski
print('zbior deweloperski') 
rdev = pd.read_csv(os.path.join('dev-0', 'in.tsv'), sep='\t', names=["date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])
rdev = pd.DataFrame(rdev,columns = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])

rdev_expected = pd.read_csv(os.path.join('dev-0', 'expected.tsv'), sep='\t', names=['y'])

print('dokładność modelu Zero rule na zestawie deweloperskim',
       1 - sum(rdev_expected['y']) / len(rdev))
print('-' * 100)

print('dokładność modelu dla danych deweloperskich (full model):', end = '')
print(sum(lr_full.predict(rdev) == rdev_expected['y'] ) / len(rdev_expected))

print('True Positives: ', end ='')
print(sum((lr_full.predict(rdev) == rdev_expected['y']) & (lr_full.predict(rdev) == 1)))
print('True Negatives: ', end ='')
print(sum((lr_full.predict(rdev) == rdev_expected['y']) & (lr_full.predict(rdev) == 0)))
print('False Positives: ', end ='')
print(sum((lr_full.predict(rdev) != rdev_expected['y']) & (lr_full.predict(rdev) == 1)))
print('False Negatives: ', end ='')
print(sum((lr_full.predict(rdev) != rdev_expected['y']) & (lr_full.predict(rdev) == 0)))
print('-'*100)
print('-'*100)
print('-'*100)
  
print('-'*100)
print('zapisywanie do pliku dev-0/out.tsv')
file = open(os.path.join('dev-0', 'out.tsv'), 'w')
for line in list(lr_full.predict(rdev)):
    file.write(str(line)+'\n')
print('-'*100)


#zbior testowy
print('zbior testowy')
test = pd.read_csv(os.path.join('test-A', 'in.tsv'), sep='\t', names=["date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])
test = pd.DataFrame(test,columns = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])

ol = lr_full.predict(test) 
file = open(os.path.join('test-A', 'out.tsv'), 'w')
for line in list(ol):
    file.write(str(line)+'\n') 
print('zapisano do pliku test-A/out.tsv')

print('Generowanie wykresu...')
sns.regplot(x=rdev.CO2, y=rdev_expected.y, logistic=True, y_jitter=.1)
plt.show()
