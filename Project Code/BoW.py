# import torch
# import torch.nn as nn
# import torch.functional as F
import csv
import nltk
nltk.download('stopwords')
import re
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectPercentile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torchmetrics.classification import BinaryAccuracy
import os
from sklearn.utils import shuffle
from scipy.sparse import coo_matrix

print(device)


class Classifier(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.hidden1 = nn.Linear(hidden, hidden*2)
        self.act1 = nn.Sigmoid()
        self.hidden2 = nn.Linear(hidden*2, hidden)
        self.act2 = nn.Sigmoid()
        self.hidden3 = nn.Linear(hidden, 250)
        self.act3 = nn.Sigmoid()
        self.output = nn.Linear(250,1)
        self.act_output = nn.Sigmoid()
 
    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.act_output(self.output(x))
        return x
    
if __name__ == "__main__":
    sents = []
    scores = []
    dirlist = os.listdir('Training/')
    for item in dirlist:

        with open('Training/'+item) as file:
            reader = csv.reader(file, delimiter=',')
            next(reader, None)
            for i in reader:
                sents.append(i[0])
                scores.append(i[1])

    dirlist = os.listdir('Test/')
    for item in dirlist:

        with open('Test/'+item) as file:
            reader = csv.reader(file, delimiter=',')
            next(reader, None)
            for i in reader:
                sents.append(i[0])
                scores.append(i[1])
    dirlist = os.listdir('Dev/')
    for item in dirlist:

        with open('Dev/'+item) as file:
            reader = csv.reader(file, delimiter=',')
            next(reader, None)
            for i in reader:
                sents.append(i[0])
                scores.append(i[1])
   

    for i in range(len(sents)):
        ir = []
        tok = re.findall(r"[\w']+", sents[i])
        for w in tok:
            if not w.isnumeric() and w not in stop_words:
                ir.append(w.lower())
        ir = re.sub(r'\W', ' ', ' '.join(ir))
        ir = re.sub("[\d]+", '', ir)
        sents[i] = ir

    
    Y = []

    for i in scores:
        if i == "1.0":
            Y.append(0)
        else:
            Y.append(1)
    

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sents)
    print(X.shape)
    selector = SelectPercentile(percentile=10).fit(X,Y)
    X = selector.fit_transform(X, Y)

    

    X_test = X[36000:46000]
    Y_test = Y[36000:46000]
  
    X_val = X[46000:]
    Y_val = Y[46000:]

    X = X[0:36000]
    Y = Y[0:36000]

    # print(X.shape)
    # print(X_test.shape)

    X_sparse = coo_matrix(X)
    X, X_sparse, Y = shuffle(X, X_sparse, Y, random_state=0)
    
    
    # print(len(X.toarray().shape))
    inputsize = len(X.toarray()[0])

    X = torch.tensor(X.toarray(), dtype=torch.float32).to(device)
    Y = torch.tensor(Y, dtype=torch.float32)
    Y = Y.view(len(Y),1).to(device)

    X_test = torch.tensor(X_test.toarray(), dtype=torch.float32).to(device)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)
    Y_test = Y_test.view(len(Y_test),1).to(device)

    X_val = torch.tensor(X_val.toarray(), dtype=torch.float32).to(device)
    Y_val = torch.tensor(Y_val, dtype=torch.float32)
    Y_val = Y_val.view(len(Y_val),1).to(device)

    # X_test = torch.from_numpy(X_test).to(device)
    # Y_test = torch.from_numpy(Y_test)
    # Y_test = Y_test.view(len(Y_test),1).to(device)

    # print(X.shape)
    # print(X_test.shape)

    model = Classifier(inputsize).to(device)
    print(model)

    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 10
    batch_size = 100
    metric = BinaryAccuracy().to(device)
    for epoch in range(n_epochs):
        acc = []
        y_pred = 0
        
        # metric.attach(default_evaluator, "accuracy")
        for i in range(0, len(X), batch_size):
            Xbatch = X[i:i+batch_size]
            y_pred = model(Xbatch)
            ybatch = Y[i:i+batch_size]
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # state = default_evaluator.run([[y_pred, y_true]])
            # if i == len(X)-batch_size:
            #     print(y_pred.view(1,10), ybatch.view(1,10))
            #     print(metric(y_pred, ybatch))
            
            acc.append(metric(y_pred, ybatch).cpu())
            #print(metric(y_pred, ybatch))
        print("accuracy:",np.average(acc))
        # print(y_pred)
        print(f'Finished epoch {epoch}, latest loss {loss}')


    y_pred = model(X_val)
    print('validation accuracy:',metric(y_pred, Y_val))

    y_pred = model(X_test)
    print('testing accuracy:',metric(y_pred, Y_test))










