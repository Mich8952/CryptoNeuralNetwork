import pandas as pd
import tensorflow as tf
from NeuralClass import NeuralModel

fileName = input("Enter the file name excluding extensions (only .csv are supported as of right now) : ")

Model = NeuralModel(fileName, 11, 64, 1, 60, 0.90)

uIn = input("Train, Evalulate, CSV? : ")
if uIn == "Train" or uIn == 'train':
    Model.train()
elif uIn == "Evaluate" or uIn == "evaluate" or uIn == "E":
    Model.predictScore()
else:
    Model.predictToCSV()


    #setup predict to allow the user to specify the model to use from the folder which can be taken as a parameter into the predict
    #also give default values for the initiliter stuff


#MAIN
#df = pd.read_csv("data10001.csv")
#dataframeName = 'data10001'
#EPOCHS = 10
#BATCH_SIZE = 64
#futurePeriods = 3
#seqLen = 60
#percentSplitMain = 0.95

##LAST FUNCTION THAT OUPUTS THE ARRAY OF PREDICTION FUNCTION DOESNT WORK PROPERLY. The actual accuracy output is correct but the method of concatenating and turning the 3d array into apandas dataframe has errors. The data, predict labels, and true labels are unsynced