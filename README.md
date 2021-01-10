# CryptoNeuralNetwork
IN VERY EARLY STAGES
Work in progress project that uses neural networks to predict when prices either increase or decrease.

Current: Can train using CSV data that includes Close and Volume data for crypto currencies. Currently supports multiple moving averages, MACD, OBV, RSI, and Bollinger Bands.
Full experimentation with all indicators and other factors chosen in the neural network have yet to be extensively explored. Current profile 2 configurations achieve ~78% accuracy across training and validation data. New data passed also provides ~78% accuracy.

Next steps:
- Experiment with different values and profiles for the neural network (i.e. different activation functions, # of neurons, # of layers, learning rate, decay rate, etc.
- Implement real time predictions using deque and coinmarketcap API
- Experiment with including more indicators and excluding others
- Continue to work on the predictToCSV() function in order to export the data to a CSV file properly (currently not working: rows are not consistent (suspecting issues with matching as ptc change was applied after labels were created (may have to redefine the label function to do it based upon ptc change)))
- Use an exchange's API to automate buys & sells under certain loss and gain conditions for the purpose of profiting.
- Add a ML algorithm that uses google trends API to factor in popularities of cryptocurrencies in google searches into the prediction of the model
- Naming conventions are a bit inconsistent and will be fixed
