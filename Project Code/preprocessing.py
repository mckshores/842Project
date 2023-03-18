import json
import datetime
import pandas
from collections import OrderedDict
import csv
import glob
import os

import pandas as pd


def preProcess():

    #I tried doing this all at once in the comment block below but it took too long
    #I'd recommend doing the categories one at a time then executing combineCategories()
    preProcessCategory("FILE_NAME_HERE.json")

""""
    tempTime = datetime.datetime.now()
    testDF1, devDF1, trainDF1, topPosWords1, topNegWords1 = preProcessCategory("Arts_Crafts_and_Sewing.json")
    print("Finished Arts and Crafts. Duration in seconds: ", (datetime.datetime.now()-tempTime).seconds)

    tempTime = datetime.datetime.now()
    testDF2, devDF2, trainDF2, topPosWords2, topNegWords2 = preProcessCategory("Automotive.json")
    print("Finished Automotive. Duration in seconds: ", (datetime.datetime.now() - tempTime).seconds)

    tempTime = datetime.datetime.now()
    testDF3, devDF3, trainDF3, topPosWords3, topNegWords3 = preProcessCategory("Movies_and_TV.json")
    print("Finished Movies_and_TV. Duration in seconds: ", (datetime.datetime.now() - tempTime).seconds)

    tempTime = datetime.datetime.now()
    testDF4, devDF4, trainDF4, topPosWords4, topNegWords4 = preProcessCategory("Luxury_Beauty.json")
    print("Finished Luxury_Beauty. Duration in seconds: ", (datetime.datetime.now() - tempTime).seconds)

    tempTime = datetime.datetime.now()
    testDF5, devDF5, trainDF5, topPosWords5, topNegWords5 = preProcessCategory("Video_Games.json")
    print("Finished Video_Games. Duration in seconds: ", (datetime.datetime.now() - tempTime).seconds)

    tempTime = datetime.datetime.now()
    testDF6, devDF6, trainDF6, topPosWords6, topNegWords6 = preProcessCategory("Gift_Cards.json")
    print("Finished Gift_Cards. Duration in seconds: ", (datetime.datetime.now() - tempTime).seconds)

    tempTime = datetime.datetime.now()
    testDF7, devDF7, trainDF7, topPosWords7, topNegWords7 = preProcessCategory("Office_Products.json")
    print("Finished Office_Products. Duration in seconds: ", (datetime.datetime.now() - tempTime).seconds)

    tempTime = datetime.datetime.now()
    testDF8, devDF8, trainDF8, topPosWords8, topNegWords8 = preProcessCategory("Sports_and_Outdoors.json")
    print("Finished Sports_and_Outdoors. Duration in seconds: ", (datetime.datetime.now() - tempTime).seconds)

    tempTime = datetime.datetime.now()
    testDF9, devDF9, trainDF9, topPosWords9, topNegWords9 = preProcessCategory("Tools_and_Home_Improvement.json")
    print("Finished Tools_and_Home_Improvement. Duration in seconds: ", (datetime.datetime.now() - tempTime).seconds)

    tempTime = datetime.datetime.now()
    testDF10, devDF10, trainDF10, topPosWords10, topNegWords10 = preProcessCategory("Toys_and_Games.json")
    print("Finished Toys_and_Games. Duration in seconds: ", (datetime.datetime.now() - tempTime).seconds)

    testFrames = [testDF1, testDF2, testDF3, testDF4, testDF5, testDF6, testDF7, testDF8, testDF9, testDF10]
    testResult = pandas.concat(testFrames)
    testResult.to_csv("testData.csv", sep=',', index=False)

    devFrames = [devDF1, devDF2, devDF3, devDF4, devDF5, devDF6, devDF7, devDF8, devDF9, devDF10]
    devResult = pandas.concat(devFrames)
    devResult.to_csv("devData.csv", sep=',', index=False)

    trainFrames = [trainDF1, trainDF2, trainDF3, trainDF4, trainDF5, trainDF6, trainDF7, trainDF8, trainDF9, trainDF10]
    trainResult = pandas.concat(trainFrames)
    trainResult.to_csv("trainData.csv", sep=',', index=False)
"""

def combineCategories():
    testFiles = os.listdir('Test')
    testFrames = []
    for file in testFiles:
        df = pd.read_csv("Test/" + file)
        testFrames.append(df)
    testResult = pandas.concat(testFrames)
    testResult.to_csv("testData.csv", sep=',', index=False)

    devFiles = os.listdir('Dev')
    devFrames = []
    for file in devFiles:
        df = pd.read_csv("Dev/" + file)
        devFrames.append(df)
    devResult = pandas.concat(devFrames)
    devResult.to_csv("devData.csv", sep=',', index=False)
    combineTopWords()

def combineTopWords():
    posFiles = os.listdir('PosTopWords')
    negFiles = os.listdir('NegTopWords')
    posDict = {}
    negDict = {}
    for file in posFiles:
        df = pd.read_csv("PosTopWords/" + file, header=None)
        for i in range(0, len(df)):
            word = df.iat[i,0]
            if word in posDict.keys():
                posDict[word] = posDict[word] + 1
            else:
                entry = {word: 1}
                posDict.update(entry)

    for file in negFiles:
        df = pd.read_csv("NegTopWords/" + file, header=None)
        for i in range(0, len(df)):
            word = df.iat[i,0]
            if word in negDict.keys():
                negDict[word] = negDict[word] + 1
            else:
                entry = {word: 1}
                negDict.update(entry)

    posDict = sorted(posDict.items(), key=lambda x: x[1], reverse=True)
    with open("TopPosWords.csv", 'w') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in posDict:
            writer.writerow([key, value])

    negDict = sorted(negDict.items(), key=lambda x: x[1], reverse=True)
    with open("TopNegWords.csv", 'w') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in negDict:
            writer.writerow([key, value])


def preProcessCategory(filename):
    testData = []
    devData = []
    trainData = []

    # Iterate thru each category
    data = []

    for line in open("Reviews/" + filename, "r"):
        data.append(json.loads(line))
    posReviews, negReviews = populateLists(data)

    # Take 20% off the top of each list and add to test data
    posTestData = posReviews[:500]
    posTrainData = posReviews[500:]
    negTestData = negReviews[:500]
    negTrainData = negReviews[500:]
    testData.append(posTestData)
    testData.append(negTestData)
    # Take 10% off the top of each list and add to dev data
    devData.append(posTrainData[:200])
    devData.append(negTrainData[:200])

    posTrainData = posTrainData[200:]
    negTrainData = negTrainData[200:]

    # write dev data to a csv, also returning the reviews as a DataFrame
    devName = "Dev/" + filename.split(".")[0] + "_dev.csv"
    devDataFrame = castToDf(devData, devName)

    # write test data to a csv, also returning the reviews as a DataFrame
    testName = "Test/" + filename.split(".")[0] + "_test.csv"
    testDataFrame = castToDf(testData, testName)

    # Search thru pos list to get top words
    # Write words in order to csv
    topPosWords = findTopWords(posTrainData, "PosTopWords/" + filename.split(".")[0] + "_Pos.csv")
    # Search thru neg list to get top words
    # Write words in order to csv
    topNegWords = findTopWords(negTrainData, "NegTopWords/" + filename.split(".")[0] + "_Neg.csv")

    # Combine pos/neg training data and write to csv
    trainData.append(posTrainData)
    trainData.append(negTrainData)
    trainName = "Training/" + filename.split(".")[0] + "_train.csv"
    trainDataFrame = castToDf(trainData, trainName)

    return testDataFrame,devDataFrame,trainDataFrame,topPosWords, topNegWords

def populateLists(json):
    posList = []
    negList = []
    i = 0
    reviewerID = 'Amazon Customer'
    while(len(posList) < 2500 or len(negList) < 2500):
        review = json[i]
        keys = "reviewerName" in review.keys() and "overall" in review.keys() and "reviewText" in review.keys()
        #Check that all the keys exist and the name is not Amazon Customer
        #OPTIONAL: Add ReviewerName when appending to lists
        if keys and review["reviewerName"] != reviewerID:
            if review["overall"] == 5.0 and len(posList) < 2500:
                posList.append((review["reviewText"], review["overall"]))
            elif review["overall"] == 1.0 and len(negList) < 2500:
                negList.append((review["reviewText"], review["overall"]))
        i += 1
    return posList, negList

def castToDf(data, filename):
    reviewList = []
    ratingList = []
    for set in data:
        for review in set:
            reviewList.append(review[0])
            ratingList.append(review[1])

    df = pandas.DataFrame(data={"Review": reviewList, "Rating": ratingList})
    df.to_csv(filename, sep=',', index=False)

    return df

#Currently this has a manual step of sifting out meaningless words.
#Strech goal is to automate this with a POS tagger
#We need to look at the list and talk about what kinds of words we want to keep
def findTopWords(reviews, filename):
    dictionary = {}
    for r in reviews:
        wordList = r[0].split()
        for w in wordList:
            if w in dictionary.keys():
                dictionary[w] = dictionary[w] + 1
            else:
                entry = {w: 1}
                dictionary.update(entry)
    #sort dictionary most to least popular
    dictionary = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in dictionary:
            writer.writerow([key, value])

