import sklearn_crfsuite
from nltk.corpus import treebank
import pandas
from nltk.tag.util import untag
import os


def CRF():

    #Setup training data
    sentences = treebank.tagged_sents(tagset='universal')

    X_train, y_train = transform_to_dataset(sentences)

    #Train CRF Model
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.25,
        c2=0.3,
        max_iterations=100,
        all_possible_transitions=True
    )

    crf.fit(X_train, y_train)

    #Setup testing data
    test_sentences = []
    testFiles = os.listdir('Neutral/Training')
    for file in testFiles:
        df = pandas.read_csv("Neutral/Training/" + file)
        temp = df['Review'].tolist()
        for t in temp:
            test_sentences.append(t)

        X_test = transform_test_to_dataset(test_sentences)
        #Run prediction
        y_pred = crf.predict(X_test)
        #Write to CSV
        df = pandas.DataFrame(data={"Word": X_test, "POS": y_pred})
        df.head(100).to_csv("POS/Neutral/" + file.split(".")[0] + "_pos.csv", sep=',', index=False)

def split_data(sentences):
    X,y = [], []
    for pos_tags in sentences:
        for index, (word, pos) in enumerate(pos_tags):
            X.append(create_Features(remove_tag(pos_tags), index))
            y.append(pos)
    return X,y

def create_Features(sentence, index):
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }

def remove_tag(sentence):
    #return a list of the tags
    return [w for w, _ in sentence]

def transform_to_dataset(tagged_sentences):
    X, y = [], []

    for tagged in tagged_sentences:
        X.append([create_Features(untag(tagged), index) for index in range(len(tagged))])
        y.append([tag for _, tag in tagged])

    #print("Train: ", X[0])
    return X, y

def transform_test_to_dataset(sentences):
    X = []

    for s in sentences:
        words = s.split()
        sentenceWords = []
        for index in range(0, len(words)):
            #print(words[index])
            sentenceWords.append(create_Features(words, index))
        X.append(sentenceWords)
    #print("Test: ", X[0])

    return X
