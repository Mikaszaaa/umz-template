import pandas as pd
import seaborn as sns
import os 
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np


print('Zbior treningowy')

rtrain = pd.read_csv('train/in.tsv', header=None, sep='\t')
rtrain.columns=["powroty", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "twentyone", "twentytwo", "twentythree", "twentyfour", "twentyfive", "twentysix", "twentyseven", "twentyeight", "twentynine", "thirty", "thirtyone", "thirtytwo", "thirtythree", "thirtyfour"]
rtrain = pd.DataFrame(rtrain, columns = ["powroty", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "twentyone", "twentytwo", "twentythree", "twentyfour", "twentyfive", "twentysix", "twentyseven", "twentyeight", "twentynine", "thirty", "thirtyone", "thirtytwo", "thirtythree", "thirtyfour"])
print('Pierwsze wiersze tabeli:')
print(rtrain.head())

print('-'*100)

print('Opis danych')

print(rtrain.describe())

print('-'*100)

print('Grupowanie pozwalajace wybrac najlepsza zmienna')

print(rtrain.groupby('powroty').aggregate(np.sum))

print('Wybieramy zmienna five')

print('-'*100)
print('Budujemy model lr...')

lr = LogisticRegression()
lr.fit(rtrain.five.values.reshape(-1, 1), rtrain.powroty)

re = lr.predict(rtrain.five.values.reshape(-1, 1))

print('dokoladnosc modelu lr dla jednej zmiennej five na danych treningowych: ', end='')
print(sum(re == rtrain.powroty) / len(rtrain))
print('-'*100)

print('True Positives: ', end ='')
print(sum((re == rtrain.powroty) & (re == "b")))
print('True Negatives: ', end ='')
print(sum((re == rtrain.powroty) & (re == "g")))
print('False Positives: ', end ='')
print(sum((re != rtrain.powroty) & (re == "b")))
print('False Negatives: ', end ='')
print(sum((re != rtrain.powroty) & (re == "g")))

print('-'*100)


print('Zbior deweloperski')

rdev = pd.read_csv(os.path.join('dev-0', 'in.tsv'), sep='\t', names=["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "twentyone", "twentytwo", "twentythree", "twentyfour", "twentyfive", "twentysix", "twentyseven", "twentyeight", "twentynine", "thirty", "thirtyone", "thirtytwo", "thirtythree", "thirtyfour"])

rdev = pd.DataFrame(rdev,columns = ["five"])

print('Pierwsze wiersze tabeli danych deweloperskich ze zmienna five')
print(rdev.head())

kl = lr.predict(rdev.five.values.reshape(-1, 1))


#pd.DataFrame(kl).to_csv('dev-0/out.tsv')

print('-'*100)
print('zapisywanie do pliku dev-0/out.tsv...')
file = open(os.path.join('dev-0', 'out.tsv'), 'w')
for line in list(kl):
    file.write(str(line)+'\n')
print('-'*100)

rdev_expected = pd.read_csv('dev-0/expected.tsv', sep='\t', header=None, names=['powroty'])

print('Pierwsze wiersze danych expected')
rdev_expected.head()


print('dokładność modelu dla danych deweloperskich:', end = '')
print(sum(kl == rdev_expected['powroty']) / len(rdev))


print('True Positives: ', end ='')
print(sum((kl == rdev_expected['powroty']) & (kl == "b")))
print('True Negatives: ', end ='')
print(sum((kl == rdev_expected['powroty']) & (kl == "g")))
print('False Positives: ', end ='')
print(sum((kl != rdev_expected['powroty']) & (kl == "b")))
print('False Negatives: ', end ='')
print(sum((kl != rdev_expected['powroty']) & (kl == "g")))

print('-'*100)


print('Zbior testowy')

ui = pd.read_csv('test-A/in.tsv', sep='\t', names=["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "twentyone", "twentytwo", "twentythree", "twentyfour", "twentyfive", "twentysix", "twentyseven", "twentyeight", "twentynine", "thirty", "thirtyone", "thirtytwo", "thirtythree", "thirtyfour"])
print(ui.head())


er = lr.predict(pd.DataFrame(ui, columns=['five']))


#pd.DataFrame(er).to_csv('test-A/out.tsv')

file = open(os.path.join('test-A', 'out.tsv'), 'w')
for line in list(er):
    file.write(str(line)+'\n') 
print('zapisano do pliku test-A/out.tsv')

#print('Generowanie wykresu...')
#sns.regplot(x=rdev.five, y=rdev_expected.powroty, logistic=True, y_jitter=.1)
#plt.show()




