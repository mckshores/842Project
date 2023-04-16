import pandas as pd
import json

def process():

    data = []
    for line in open('image_review_all.json', "r"):
        data.append(json.loads(line))

    newData = []
    for d in data:
        newData.append((d["review_text"], d["rating"]))

    reviewList = []
    ratingList = []
    for review in newData:
        reviewList.append(review[0])
        ratingList.append(review[1])

    df = pd.DataFrame(data={"Review": reviewList, "Rating": ratingList})
    df.to_csv("GoogleReviews.csv", sep=',', index=False, escapechar='\\')