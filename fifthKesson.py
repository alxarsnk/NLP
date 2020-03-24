import pymorphy2
import pandas
from string import punctuation
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
morph = pymorphy2.MorphAnalyzer()

# считали из файла
df = pandas.read_csv("/Users/alxarsnk/Downloads/Отзывы кино - Лист1.csv", encoding="utf-8")

# тестовые данные
valid = df["title"].isin(['Интерстеллар', "Омерзительная восьмерка", "Тройной форсаж: Токийский дрифт"])
test = df[valid]
del test['title']

# удалили наши данные(оставили данные для обучения)
df = df.loc[df['title'] != "Интерстеллар"]
df = df.loc[df['title'] != "Омерзительная восьмерка"]
df = df.loc[df['title'] != "Тройной форсаж: Токийский дрифт"]
del df['title']

allWordsRead = open("AllWords.txt", "r")
allWordsList = allWordsRead.read().split("\n")
uniqWords = set(allWordsList)
allWordsRead.close()

def tokenizeWord(word):
    word = tokenizer.tokenize(word)
    if len(word) == 1:
        word = word[0]
    else:
        word = ""
    return word

for word in uniqWords:
    word = tokenizeWord(word)


def getPosArr(words):
    verbcount = 0
    nouncount = 0
    adjcount = 0
    advcount = 0
    for word in words:
        word = tokenizeWord(morph.parse(word)[0].normal_form)
        pos = morph.parse(word)[0].tag.POS
        if (pos == "ADJF") | (pos == "ADJS"):
            adjcount += 1
        elif pos == "NOUN":
            nouncount += 1
        elif (pos == "VERB") | (pos == "INFN"):
            verbcount += 1
        elif pos == "ADVB":
            advcount += 1
    return [verbcount, nouncount, adjcount, advcount]


def getBagOfWords(df, uniqWords):
    index = 0
    resDF = pandas.DataFrame(columns=uniqWords)
    answerDF = pandas.DataFrame(columns=["Answer"])
    for tuple in df.values:
        bagDict = dict.fromkeys(uniqWords, 0)
        index += 1
        print(index)
        words = tuple[0].split()
        for word in words:
            wordNF = morph.parse(word)[0].normal_form
            bagDict[wordNF] += 1
        resDF.loc[index] = list(bagDict.values())
        answerDF.loc[index] = tuple[1]
    return resDF, answerDF

def getPunctArr(text):
    punctList = list(punctuation)
    dictionary = dict.fromkeys(punctList, 0)
    charList = list(text)
    for char in charList:
        if char in punctList:
            dictionary[char] += 1
    return list(dictionary.values())

studyDFs = getBagOfWords(df, uniqWords)
testDFs = getBagOfWords(test, uniqWords)

print(studyDFs[0].shape, studyDFs[1].shape)

X = studyDFs[0].to_numpy()
y = studyDFs[1]["Answer"]

linearCLF = LinearSVC(random_state=0).fit(X, y)
randomForestCLF = RandomForestClassifier(max_depth=20, random_state=0).fit(X, y)
logisticCLF = LogisticRegression(random_state=0).fit(X, y)

linearPredict = linearCLF.predict(testDFs[0])
randomForestPredict = randomForestCLF.predict(testDFs[0])
logisticPredict = logisticCLF.predict(testDFs[0])

linearMetrics = precision_recall_fscore_support(testDFs[1]["Answer"].values, linearPredict, average='weighted')
randomForestMetric = precision_recall_fscore_support(testDFs[1]["Answer"].values, randomForestPredict, average='weighted')
logisticMetrics = precision_recall_fscore_support(testDFs[1]["Answer"].values, logisticPredict, average='weighted')

print(linearMetrics)
print(randomForestMetric)
print(logisticMetrics)


allColumns = list(studyDFs[0].columns)+["verb","noun","adjv","advb"]+list(punctuation)
bowColumns = ["verb","noun","adjv","advb"]+list(punctuation)
posColumns = list(studyDFs[0].columns)+list(punctuation)
punctColumns = list(studyDFs[0].columns)+["verb","noun","adjv","advb"]

all_features_df = pandas.DataFrame(columns=allColumns)
all_features_bow_df = pandas.DataFrame(columns=bowColumns)
all_features_pos_df = pandas.DataFrame(columns=posColumns)
all_features_punct_df = pandas.DataFrame(columns=punctColumns)

index = 1
for index in range(len(studyDFs[0].index)):
    print(index)

    bow = list(studyDFs[0].to_numpy()[index])
    pos = list(getPosArr(df.values[index][0].split()))
    punct = getPunctArr(df.values[index][0])

    all_features = bow + pos + punct
    all_features_bow = pos + punct
    all_features_pos = bow + punct
    all_features_punct = bow + pos

    all_features_df.loc[index] = list(all_features)
    all_features_bow_df.loc[index] = all_features_bow
    all_features_pos_df.loc[index] = all_features_pos
    all_features_punct_df.loc[index] = all_features_punct

print(all_features_df.shape, all_features_bow_df.shape, all_features_pos_df.shape, all_features_punct_df.shape)



all_features_df_test = pandas.DataFrame(columns=allColumns)
all_features_bow_df_test = pandas.DataFrame(columns=bowColumns)
all_features_pos_df_test = pandas.DataFrame(columns=posColumns)
all_features_punct_df_test = pandas.DataFrame(columns=punctColumns)

for index in range(len(testDFs[0].index)):
    print(index)
    bow = list(testDFs[0].to_numpy()[index])
    pos = list(getPosArr(test.values[index][0].split()))
    punct = getPunctArr(test.values[index][0])

    all_features = bow + pos + punct
    all_features_bow = pos + punct
    all_features_pos = bow + punct
    all_features_punct = bow + pos

    all_features_df_test.loc[index] = list(all_features)
    all_features_bow_df_test.loc[index] = all_features_bow
    all_features_pos_df_test.loc[index] = all_features_pos
    all_features_punct_df_test.loc[index] = all_features_punct

X_all = all_features_df.to_numpy()
X_bow = all_features_bow_df.to_numpy()
X_pos = all_features_pos_df.to_numpy()
X_punct = all_features_punct_df.to_numpy()

randomForestCLF_all = RandomForestClassifier(max_depth=20, random_state=0).fit(X_all, y)
randomForestCLF_bow = RandomForestClassifier(max_depth=20, random_state=0).fit(X_bow, y)
randomForestCLF_pos = RandomForestClassifier(max_depth=20, random_state=0).fit(X_pos, y)
randomForestCLF_punct = RandomForestClassifier(max_depth=20, random_state=0).fit(X_punct, y)

randomForestPredict_all = randomForestCLF_all.predict(all_features_df_test)
randomForestPredict_bow = randomForestCLF_bow.predict(all_features_bow_df_test)
randomForestPredict_pos = randomForestCLF_pos.predict(all_features_pos_df_test)
randomForestPredict_punct = randomForestCLF_punct.predict(all_features_punct_df_test)

randomForestMetric_all = precision_recall_fscore_support(testDFs[1]["Answer"].values, randomForestPredict_all, average='weighted')
randomForestMetric_bow = precision_recall_fscore_support(testDFs[1]["Answer"].values, randomForestPredict_bow, average='weighted')
randomForestMetric_pos = precision_recall_fscore_support(testDFs[1]["Answer"].values, randomForestPredict_pos, average='weighted')
randomForestMetric_punct = precision_recall_fscore_support(testDFs[1]["Answer"].values, randomForestPredict_punct, average='weighted')

print(f"Модель со всеми показателями {randomForestMetric_all}")
print(f"Модель без мешка слов {randomForestMetric_bow}")
print(f"Модель без частей речи {randomForestMetric_pos}")
print(f"Модель без пунктцации {randomForestMetric_punct}")
