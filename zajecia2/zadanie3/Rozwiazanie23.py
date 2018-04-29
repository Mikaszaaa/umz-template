import pandas as pd
import seaborn as sns
import os 
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np


print('Zbior treningowy')

rtrain = pd.read_csv('train/in.tsv', header=None, sep='\t', names=["powroty", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "twentyone", "twentytwo", "twentythree", "twentyfour", "twentyfive", "twentysix", "twentyseven", "twentyeight", "twentynine", "thirty", "thirtyone", "thirtytwo", "thirtythree", "thirtyfour"])

print('Pierwsze wiersze tabeli:')
print(rtrain.head())

print('-'*100)

print('Opis danych')

print(rtrain.describe())

print('-'*100)

X = pd.DataFrame(rtrain, columns=["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "twentyone", "twentytwo", "twentythree", "twentyfour", "twentyfive", "twentysix", "twentyseven", "twentyeight", "twentynine", "thirty", "thirtyone", "thirtytwo", "thirtythree", "thirtyfour"])

print('Budujemy model lr...')
lr = LogisticRegression()
lr.fit(X, rtrain.powroty)

re = lr.predict(X)

print('dokoladnosc modelu lr dla wszystkich zmiennych na danych treningowych: ', end='')
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

rdev = pd.DataFrame(rdev,columns = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "twentyone", "twentytwo", "twentythree", "twentyfour", "twentyfive", "twentysix", "twentyseven", "twentyeight", "twentynine", "thirty", "thirtyone", "thirtytwo", "thirtythree", "thirtyfour"])
print('Pierwsze wiersze tabeli z danymi deweloperskimi')
print(rdev.head())
print('Przewidujemy powroty dla danych deweloperkich...')
kl = lr.predict(rdev)

print('-'*100)
print('zapisywanie do pliku dev-0/out.tsv...')
file = open(os.path.join('dev-0', 'out.tsv'), 'w')
for line in list(kl):
    file.write(str(line)+'\n')
print('-'*100)

rdev_expected = pd.read_csv('dev-0/expected.tsv', sep='\t', header=None, names=['powroty'])

print('Pierwsze wiersze danych expected')
print(rdev_expected.head())


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

print('Zero rule na zbiorze deweloperskim')
zero = pd.read_csv('dev-0/zeroR.tsv', sep='\t', header=None, names=['dane'])
print(sum(zero['dane'] == rdev_expected['powroty']) / len(zero))


print('Zbior testowy')

ui = pd.read_csv('test-A/in.tsv', sep='\t', names=["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "twentyone", "twentytwo", "twentythree", "twentyfour", "twentyfive", "twentysix", "twentyseven", "twentyeight", "twentynine", "thirty", "thirtyone", "thirtytwo", "thirtythree", "thirtyfour"])
print(ui.head())


er = lr.predict(pd.DataFrame(ui, columns=["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "twentyone", "twentytwo", "twentythree", "twentyfour", "twentyfive", "twentysix", "twentyseven", "twentyeight", "twentynine", "thirty", "thirtyone", "thirtytwo", "thirtythree", "thirtyfour"]))




file = open(os.path.join('test-A', 'out.tsv'), 'w')
for line in list(er):
    file.write(str(line)+'\n') 
print('zapisano do pliku test-A/out.tsv')

#print('Generowanie wykresu...')
#sns.regplot(x=rdev.five, y=rdev_expected.powroty, logistic=True, y_jitter=.1)
#plt.show()




