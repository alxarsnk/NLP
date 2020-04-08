import pandas
import pymorphy2
from itertools import islice
import math
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
import pickle
from sklearn.metrics import classification_report
import keras_metrics



def writeToFile(dictionary):
    with open('someDict.txt', 'wb') as out:
        pickle.dump(dictionary, out)

def readDictFromFile():
    with open('someDict.txt', 'rb') as inp:
        d_in = pickle.load(inp)
    return  d_in
morph = pymorphy2.MorphAnalyzer()

df = pandas.read_csv("/Users/alxarsnk/Downloads/мой нлп - Лист1-2.csv", encoding="utf-8")

# тестовые данные
valid = df["title"].isin(['Интерстеллар', "Омерзительная восьмерка", "Тройной форсаж: Токийский дрифт"])
test = df[valid]
del test['title']

# удалили наши данные(оставили данные для обучения)
df = df.loc[df['title'] != "Интерстеллар"]
df = df.loc[df['title'] != "Омерзительная восьмерка"]
df = df.loc[df['title'] != "Тройной форсаж: Токийский дрифт"]
del df['title']

def take(n, iterable):
    return list(islice(iterable, n))

def get500favouriteWordsIn(df):
    allWordsRead = open("AllWords.txt", "r")
    allWordsList = allWordsRead.read().split("\n")
    uniqWords = set(allWordsList)
    allWordsRead.close()
    frequencyDict = dict.fromkeys(uniqWords, 0)
    index = 0
    for tuple in df.values:
        index+=1
        print(index)
        words = tuple[0].split()
        for word in words:
            wordNF = morph.parse(word)[0].normal_form
            frequencyDict[wordNF] += 1
    sortedDict = sorted((value, key) for (key, value) in frequencyDict.items())
    top500 = []
    for index in range(500):
        top500.append(sortedDict[-index][1])
    return top500
favouriteWords = get500favouriteWordsIn(df)

def getTF_forComment(words):
    frequncyDict = dict.fromkeys(favouriteWords, 0)
    for word in words:
        wordNF = morph.parse(word)[0].normal_form
        if wordNF in favouriteWords:
            frequncyDict[wordNF] += 1
    tfDict = {}
    bagOfWordsCount = len(favouriteWords)
    for word, count in frequncyDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict.values()

def getTFIDF_forComments(comments):
    index = 0
    N = len(comments)
    idfDict = dict.fromkeys(favouriteWords, 0)
    tfDicts = []
    for comment in comments.values:
        index += 1
        print(index)
        words = comment[0].split()
        commentDict = dict.fromkeys(favouriteWords, 0)
        for word in words:
            wordNF = morph.parse(word)[0].normal_form
            if wordNF in favouriteWords:
                commentDict[wordNF] += 1
        tfDict = {}
        bagOfWordsCount = len(favouriteWords)
        for word, count in commentDict.items():
            tfDict[word] = count / float(bagOfWordsCount)
        tfDicts.append(tfDict)
        for word, val in commentDict.items():
            if val > 0:
                idfDict[word] += 1
    for word, val in idfDict.items():
        if val > 0:
            idfDict[word] = math.log(N / float(val))
    tfIDFs = pandas.DataFrame(columns=favouriteWords)
    for ind,tfDict in enumerate(tfDicts):
        tfidf = {}
        print(ind)
        for word, val in tfDict.items():
            tfidf[word] = val * idfDict[word]
        tfIDFs.loc[ind] = list(tfidf.values())
    return tfIDFs


def getX_model(df):
    tfidfDF = getTFIDF_forComments(df)
    return tfidfDF

x_train = getX_model(df).values
x_test = getX_model(test).values

y_train = df["label"].values
y_test = test["label"].values

num_classes = 3

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = Sequential()

model.add(Dense(512, input_shape=(499,)))
model.add(Dense(3))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=128)
predictions = model.predict(x_test)


print(predictions)
print(y_test)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1)))

