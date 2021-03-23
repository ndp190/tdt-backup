import pandas as pd
import sys
import pickle
import os

def azureml_main(dataframe1 = None, dataframe2 = None):
    sys.path.insert(0,".\Script Bundle")
    os.listdir(".\Script Bundle")
    model = pickle.load(open(".\Script Bundle\car-model.pkl", 'rb'))
    pred = model.predict(dataframe1)
    return pd.DataFrame([pred[0]])

