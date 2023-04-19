import json
import pandas
import csv
import os

import pandas as pd


def preProcess():

    preProcessCategory("Arts_Crafts_and_Sewing.json")
    preProcessCategory("Automotive.json")
    preProcessCategory("Luxury_Beauty.json")
    preProcessCategory("Movies_and_TV.json")
    preProcessCategory("Video_Games.json")
    preProcessCategory("Gift_Cards.json")
    preProcessCategory("Office_Products.json")
    preProcessCategory("Sports_and_Outdoors.json")
    preProcessCategory("Tools_and_Home_Improvement.json")
    preProcessCategory("Toys_and_Games.json")


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

def preProcessCategory(filename):
    testData = []
    devData = []
    trainData = []

    # Iterate thru each category
    data = []

    for line in open("Reviews/" + filename, "r"):
        data.append(json.loads(line))
    twostar, threestar, fourstar = populateLists(data)

    # Take 20% off the top of each list and add to test data
    twostarTestData = twostar[:500]
    twostarTrainData = twostar[500:]
    threestarTestData = threestar[:500]
    threestarTrainData = threestar[500:]
    fourstarTestData = fourstar[:500]
    fourstarTrainData = fourstar[500:]
    testData.append(twostarTestData)
    testData.append(threestarTestData)
    testData.append(fourstarTestData)
    # Take 10% off the top of each list and add to dev data
    devData.append(twostarTrainData[:200])
    devData.append(threestarTrainData[:200])
    devData.append(fourstarTrainData[:200])

    twostarTrainData = twostarTrainData[200:]
    threestarTrainData = threestarTrainData[200:]
    fourstarTrainData = fourstarTrainData[200:]

    # write dev data to a csv, also returning the reviews as a DataFrame
    devName = "Neutral/Dev/" + filename.split(".")[0] + "_neutral_dev.csv"
    devDataFrame = castToDf(devData, devName)

    # write test data to a csv, also returning the reviews as a DataFrame
    testName = "Neutral/Test/" + filename.split(".")[0] + "_neutral_test.csv"
    testDataFrame = castToDf(testData, testName)

    # Combine pos/neg training data and write to csv
    trainData.append(twostarTrainData)
    trainData.append(threestarTrainData)
    trainData.append(fourstarTrainData)
    trainName = "Neutral/Training/" + filename.split(".")[0] + "_neutral_train.csv"
    trainDataFrame = castToDf(trainData, trainName)

    return testDataFrame,devDataFrame,trainDataFrame

def populateLists(json):
    twostar = []
    threestar = []
    fourstar = []
    i = 0
    reviewerID = 'Amazon Customer'
    while(len(json) > i and (len(twostar) < 7500 or len(threestar) < 7500 or len(fourstar) < 7500)):
        review = json[i]
        keys = "reviewerName" in review.keys() and "overall" in review.keys() and "reviewText" in review.keys()
        #Check that all the keys exist and the name is not Amazon Customer
        #OPTIONAL: Add ReviewerName when appending to lists
        if keys and review["reviewerName"] != reviewerID:
            if review["overall"] == 2.0 and len(twostar) < 2500:
                twostar.append((review["reviewText"], review["overall"]))
            elif review["overall"] == 3.0 and len(threestar) < 2500:
                threestar.append((review["reviewText"], review["overall"]))
            elif review["overall"] == 4.0 and len(fourstar) < 2500:
                fourstar.append((review["reviewText"], review["overall"]))
        i += 1
    return twostar, threestar, fourstar

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


