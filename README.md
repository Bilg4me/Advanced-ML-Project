# Advanced-ML-Project
ML Academic Project applied to the Kaggle Challenge : Jane Street Real-Time Market Data Forecasting 2024.


The goal is to compare two ML models on the same prediction task, which is to predict one of the responders up to 6 months in the future. The models tested are Temporal Fusion Transformer (TFT) and Variational Autoencoders (VAE). 

To make the code work, you first need to download the data frome Kaggle [here](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/data) and put the parquets inside a raw_data/train_parquet folder. Then you'll be able to launch the code/preprocessing.py to create preprocessed training and validation datasets that we will use in the notebook.