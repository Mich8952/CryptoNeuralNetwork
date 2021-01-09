import random
import pandas as pd
from collections import deque
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn import preprocessing


class NeuralModel:
    dataframeName = ''
    EPOCHS = 0
    BATCH_SIZE = 0
    futurePeriods = 0
    seqLen = 0
    percentSplitMain = 0
    df = pd.DataFrame()
    NAME = f"{seqLen}-SEQ-{futurePeriods}-PRED-{int(time.time())}"

    def __init__(self,dataframeName,EPOCHS,BATCH_SIZE,futurePeriods,seqLen,percentSplitMain):
        self.dataframeName = dataframeName
        self.EPOCHS = EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
        self.futurePeriods = futurePeriods
        self.seqLen = seqLen
        self.percentSplitMain = percentSplitMain

        self.df = pd.read_csv(f"{dataframeName}.csv", na_values=['#VALUE!', '#DIV/0!'])
        self.NAME = f"{self.seqLen}-SEQ-{self.futurePeriods}-PRED-{int(time.time())}"

    def train(self):
        self.df = self.df[['Timestamp','Open','High','Low','Close','Volume_(BTC)']]
        self.df.set_index("Timestamp", inplace = True)
        self.df.dropna(inplace = True)

        #Load indicators 
        self.df = NeuralModel.ChangeToPercentCV(self.df)
        self.df = NeuralModel.MA(self.df)
        self.df = NeuralModel.MACD(self.df)
        self.df = NeuralModel.RSI(self.df)
        self.df = NeuralModel.getOBV(self.df)
        self.df = NeuralModel.BW(self.df)

        df1 = NeuralModel.alterDataFrame2(self.df, self.futurePeriods)
        trainDf, testDf = NeuralModel.splitDataByPercent(df1,self.percentSplitMain)
        train_x, train_y = NeuralModel.preProcessDataframe(trainDf, self.futurePeriods, self.seqLen)
        test_x, test_y = NeuralModel.preProcessDataframe(testDf, self.futurePeriods, self.seqLen)
        NeuralModel.statistics(train_x,train_y,test_x,test_y)
        NeuralModel.SimulateProfile2(self, train_x, train_y, test_x, test_y)

    def min_max_normalize(lst):
      minimum = min(lst)
      maximum = max(lst)
      normalized = []

      for value in lst:
        normalized_num = (value - minimum) / (maximum - minimum)
        normalized.append(normalized_num) 
        if normalized_num == None or normalized_num == 'nan': #debugging
            print(f"Value: {value}, Minimum : {minimum}, Maximum: {maximum}")
      return normalized

    def alterDataFrame(df,futurePeriods):
        subdf = pd.DataFrame()
        df['future'] = df["Close"].shift(-futurePeriods)
        subdf = df[['Volume_(BTC)','Close','future']]
        subdf['target'] = list(map(NeuralModel.buyOsell,subdf['Close'],subdf['future']))
        subdf.dropna(inplace = True)
        return subdf

    def alterDataFrame2(df,futurePeriods):
        #Create future prediction
        df['future'] = df["Close"].shift(-futurePeriods)
        df = df[['Volume_(BTC)','Close', 'MA','MAV','MAV2','MACD','SIGNAL', 'CHANGE', 'GAIN', 'LOSS', 'AvGain', 'AvLoss','RS', 'RSI','20MA','20STD','UB','LB','BW','OBV','future']]
        df.dropna(inplace = True)
        df['target'] = list(map(NeuralModel.buyOsell,df['Close'],df['future']))
        df.dropna(inplace = True)
        return df


    def buyOsell(current,future):
        if float(future) > float(current):
            return 1
        else:
            return 0


    def labelPred(confidence):
        if float(confidence) > 0.5:
            return 1
        else:
            return 0


    def falsePosGreaterX(confidence,actual):
        if float(confidence) > 0.90 and float(actual) == 0:
            return 1
        else:
            return 0


    def truePosGreaterX(confidence,actual):
        if float(confidence) > 0.90 and float(actual) == 1:
            return 1
        else:
            return 0


    def splitDataByPercent(df,percentSplitMain):
        df.dropna(inplace=True)
        pct = 1 - percentSplitMain 
        timeCron = sorted(df.index.values)  
        splitter = -int(pct*len(timeCron))  #finds the value at the 95% (default) / percentSplitMain value
        lastXpct = sorted(df.index.values)[splitter]  # row/index value
        testDf = df[(df.index >= lastXpct)]  # make the validation data where the index is in the last 5%
        trainDf = df[(df.index < lastXpct)]
        return trainDf, testDf


    def preProcessDataframe(df,futurePeriods,seqLen): 
        df = df.drop('future', 1) 
        df.dropna(subset = ['Close','Volume_(BTC)', 'target'], inplace = True)
        print(df.columns)
        for c in df.columns:
            if c != "target":
                df.dropna(inplace = True)
                #df[c] = NeuralModel.min_max_normalize(df[c].values)  #Min/Max
                df[c] = preprocessing.scale(df[c].values)             #SciKit learns scaler 
                df.dropna(inplace = True)
            
        df.dropna(inplace = True)
        sequential_data = []
        pastDays = deque(maxlen = seqLen) #deque essentially maintains the seqLen as it drops the first and adds to last

        for i in df.values:
            pastDays.append([j for j in i[:-1]])
            if len(pastDays) == seqLen: ##once we hit (seqLen) number of values then add it to the sequential data
                sequential_data.append([np.array(pastDays),i[-1]])  
                                                                  
        random.shuffle(sequential_data)

        posList = []
        negList = []

        for data, label in sequential_data:
            if label == 1:
                posList.append([data,label])
            else:                             
                negList.append([data,label])

        random.shuffle(posList)
        random.shuffle(negList)


        #Balancing data
        lower = min(len(posList), len(negList))  #lower gets the index of the smallest length so we can limit the lengths to be equal and get rid of any length biases (if 90% of the data is buy then the bot will just buy everytime)
        posList = posList[:lower]
        negList = negList[:lower]

        sequential_data  = posList + negList
        random.shuffle(sequential_data )

        independant = []
        dependant = []

        for data, labels in sequential_data:
            independant.append(data)
            dependant.append(labels)

        independant = np.array(independant) #only return data as an array

        return independant, dependant
        

    def PredictPreProcess(df,futurePeriods,seqLen):
        df = df.drop('future', 1) 
        df.dropna(inplace = True)
        print(df.columns)
        for c in df.columns:
            if c != "target":
                #df[c] = NeuralModel.min_max_normalize(df[c].values)  #Using MinMax norm
                df[c] = preprocessing.scale(df[c].values)             #Using Scikit learns scaler
                df.dropna(inplace = True)
            
        df.dropna(inplace = True)
        sequential_data = []
        pastDays = deque(maxlen = seqLen) #deque essentially maintains the seqLen as it drops the first and adds to last

        for i in df.values:
            pastDays.append([j for j in i[:-1]])
            if len(pastDays) == seqLen: ##once we hit (seqLen) number of values then add it to the sequential data
                sequential_data.append([np.array(pastDays),i[-1]])  #the -1 means the last index for target, now the seq data has 2 columns,
                                                                  # one for the data, and one for the labels, we need to save these separetely
        independant = []
        dependant = []

        for data, labels in sequential_data:
            independant.append(data)
            dependant.append(labels)

        independant = np.array(independant) #only return data as an array

        return independant, dependant


    def SimulateProfile1(self,train_x,train_y,test_x,test_y):  #not extensively tested, modifications needed            
        model = Sequential()
        model.add(LSTM(128, input_shape=(train_x.shape[1:]),return_sequences = True))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(LSTM(128, input_shape=(train_x.shape[1:]),return_sequences = True))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(LSTM(128, input_shape=(train_x.shape[1:])))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Dense(32, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(2,activation="softmax"))
        #model.add(Dense(2,activation='sigmoid'))

        #opt = tf.keras.optimizers.Adam(lr = 0.001, decay = 1e-6)
        opt = tf.keras.optimizers.Adam(lr = 0.1, decay = 1e-6)

        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
        )
        
        tensorboard = TensorBoard(log_dir="logs\{}".format(self.NAME))

        filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
        checkpoint = ModelCheckpoint("models\{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        validation_x = np.array(test_x)
        validation_y = np.array(test_y)

        #training
        history = model.fit(
            train_x, train_y,
            batch_size=self.BATCH_SIZE,
            epochs=self.EPOCHS,
            validation_data=(validation_x, validation_y),
            callbacks=[tensorboard, checkpoint])

    def SimulateProfile2(self,train_x,train_y,test_x,test_y):  #Achieves ~78% acc. on both traing and valdiation sets               
     
        model = Sequential()
        model.add(LSTM(256, input_shape=(train_x.shape[1:]),activation="tanh",return_sequences = True))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(LSTM(128,activation="tanh",return_sequences = True))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(LSTM(128, activation="tanh"))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Dense(32, activation="tanh"))
        model.add(Dropout(0.2))

        #model.add(Dense(2,activation="softmax"))
        model.add(Dense(1,activation='sigmoid'))

        #opt = tf.keras.optimizers.Adam(lr = 0.001, decay = 1e-6)
        opt = tf.keras.optimizers.Adam(lr = 0.001, decay = 1e-6)

        model.compile(
            loss='binary_crossentropy', #binary_crossentropy
            optimizer=opt,
            metrics=['accuracy']
        )
        

        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir="logs\{}".format(self.NAME),
            histogram_freq=1,
            update_freq='epoch',
            profile_batch=0
        )


        filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  
        checkpoint = ModelCheckpoint("models\{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # Saves the best
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        validation_x = np.array(test_x)
        validation_y = np.array(test_y)

        #Training
        history = model.fit(
            train_x, train_y,
            batch_size=self.BATCH_SIZE,
            epochs=self.EPOCHS,
            validation_data=(validation_x, validation_y),
            callbacks=[tensorboard, checkpoint])

        score = model.evaluate(validation_x, validation_y, verbose=0)
        print(f"Score: Test Loss --> {score[0]}, Test Accuracy --> {score[1]}")

        #print(model.get_weights()) #gets weights
        
    def ChangeToPercentCV(df):
        df = df.replace(0,np.nan)
        df['Close'] = df['Close'].pct_change() 
        df['Volume_(BTC)'] = df['Volume_(BTC)'].pct_change() 
        df.dropna(inplace = True)
        #df.to_csv("TEMPLATEOUT.csv") #for debugging
        return df

    def MA(df):   
        df['MA'] = df.rolling(window=3)['Close'].mean()
        df['MAV'] = df.rolling(window=12)['Close'].mean()
        df['MAV2'] = df.rolling(window=26)['Close'].mean()
        df.dropna(inplace = True)
        return df

    def MACD(df): 
        df['MACD'] = df['MAV'] - df['MAV2']
        df['SIGNAL'] = df.rolling(window=9)['MACD'].mean()
        df.dropna(inplace=True)
        return df

    def RSI(df):
        df['CHANGE'] = df['Close'].diff()
        df.dropna(inplace=True)

        delta = df['Close'].diff()

        window_length = 14
        gain, loss = delta.copy(), delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0

        # Calculate the EWMA
        avGain = gain.ewm(span=window_length).mean()
        avLoss = loss.abs().ewm(span=window_length).mean()


        # Calculate the RSI based on EWMA
        RS1 = avGain / avLoss
        RSI1 = 100.0 - (100.0 / (1.0 + RS1))

        # Calculate the SMA
        avGain = gain.rolling(window_length).mean()
        avLoss = loss.abs().rolling(window_length).mean()

        # Calculate the RSI based on SMA
        RS2 = avGain / avLoss
        RSI2 = 100.0 - (100.0 / (1.0 + RS2))

        df['GAIN'] = gain
        df['LOSS'] = loss
        df["AvGain"] = avGain
        df["AvLoss"] = avLoss
        df['RS'] = RS2             
        df['RSI'] = RSI2

        df = df.replace([np.inf, -np.inf], np.nan) #getting rid of inf values (ex/0)
        df.dropna(inplace=True)
        return df

    def getOBV(df):
        df['OBV'] = np.where(df['Close'] > df['Close'].shift(1), df['Volume_(BTC)'], np.where(df['Close'] < df['Close'].shift(1), -df['Volume_(BTC)'], 0)).cumsum()
        return df

    def BW(df):
        df['20MA'] = df.rolling(window=20)['Close'].mean()
        df['20STD'] = df.rolling(window=20)['Close'].std()
        df['UB'] = df['20MA'] +  df['20STD'] *2
        df['LB'] = df['20MA'] -  df['20STD'] *2
        df['BW'] = df['UB'] -  df['LB']
        df.dropna(inplace = True)
        return df

      
    def predictScore(self):
        model = tf.keras.models.load_model(r"C:\Users\micha\Desktop\BOT\mybot\pt2\PythonApplication1\models\RNN_Final-11-0.781.model") #pass this as a var eventually
        self.df = self.df[['Timestamp','Open','High','Low','Close','Volume_(BTC)']]
        self.df.set_index("Timestamp", inplace = True)
        self.df.dropna(inplace = True)
        self.df = NeuralModel.ChangeToPercentCV(self.df)
        self.df = NeuralModel.MA(self.df)
        self.df = NeuralModel.MACD(self.df)
        self.df = NeuralModel.RSI(self.df)
        self.df = NeuralModel.getOBV(self.df)
        self.df = NeuralModel.BW(self.df)
        self.df = NeuralModel.alterDataFrame2(self.df, self.futurePeriods)
        X, y = NeuralModel.PredictPreProcess(self.df, self.futurePeriods, self.seqLen)

        prediction = model.predict(X)
        score = model.evaluate(X, y, verbose=0)
        print(f"Score: Set Loss --> {score[0]}, Set Accuracy --> {score[1]}")



    def predictToCSV(self): #current not functioning properly: WIP: 
        model = tf.keras.models.load_model(r"C:\Users\micha\Desktop\BOT\mybot\pt2\PythonApplication1\models\RNN_Final-11-0.781.model") #pass this as a var eventually
        self.df = self.df[['Timestamp','Open','High','Low','Close','Volume_(BTC)']]
        self.df.set_index("Timestamp", inplace = True)
        self.df.dropna(inplace = True)
        self.df = NeuralModel.ChangeToPercentCV(self.df)
        self.df = NeuralModel.MA(self.df)
        self.df = NeuralModel.MACD(self.df)
        self.df = NeuralModel.RSI(self.df)
        self.df = NeuralModel.getOBV(self.df)
        self.df = NeuralModel.BW(self.df)
        self.df = NeuralModel.alterDataFrame2(self.df, self.futurePeriods)
        X, y = NeuralModel.PredictPreProcess(self.df, self.futurePeriods, self.seqLen)
        prediction = model.predict(X)
        ans = prediction.reshape(prediction.shape[0])
        #C = np.concatenate((X, 
                #    np.broadcast_to(np.array(ans)[:, None, None], X.shape[:-1] + (1,))), 
               #    axis = -1)

        #Debugging
        print(X.shape)
        print(prediction.shape)
        print(ans.shape)
        print(X.shape)
        #print(C.shape)
        
        newarr = X[:, 0, :]

        #Debugging
        print(newarr.shape)
        
        
        df2 = pd.DataFrame(newarr, columns = ['Volume_(BTC)','Close', 'MA','MAV','MAV2','MACD','SIGNAL', 'CHANGE', 'GAIN', 'LOSS', 'AvGain', 'AvLoss','RS', 'RSI','20MA','20STD','UB','LB','BW','OBV'])

        df2['CONFIDENCE'] = ans

        df2['PRED_CLASS'] = list(map(NeuralModel.labelPred,df2['CONFIDENCE']))

        df2["ACTUAL"] = y

        df2['FALSEPOS'] = list(map(NeuralModel.falsePosGreaterX,df2['CONFIDENCE'],df2['ACTUAL']))

        df2['TRUEPOS'] = list(map(NeuralModel.truePosGreaterX,df2['CONFIDENCE'],df2['ACTUAL']))

        #Sample output
        print(df2)

        df2.to_csv("OUT.csv") 


    def statistics(train_x, train_y, test_x, test_y):
        print(f"Lenth of Train data --> {len(train_x)} : Length of Test data --> {len(test_x)}")
        print(f"Train data: POS(Buy) --> {train_y.count(1)} : NEG(Do not buy) --> {train_y.count(0)}")
        print(f"Test data: POS(Buy) --> {test_y.count(1)} : NEG(Do not buy) --> {test_y.count(0)}")



