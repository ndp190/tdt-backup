# The script MUST contain a function named azureml_main
# which is the entry point for this module.
# imports up here can be used to 
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsRegressor

# df.
dfProcess = df[['MAKE', 'year', 'mileage', 'engine cc', 'selling_price']]
dfProcess['selling_price'] = pd.to_numeric(dfProcess['selling_price'])
data = dfProcess.values

# 1
X = data[:,0:-1]
Y = data[:,-1]
Y.ravel()

# 2
encoder = LabelEncoder()
for i in range(X.shape[1]):
    X[:,i] = encoder.fit_transform(X[:,i])

# 3
minmax = MinMaxScaler()
X = minmax.fit_transform(X)

# 4
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.9, random_state=42)

# 5
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train,Y_train)

# 6
y_pred = model.predict((X_test))

# serialize the model on disk in the special 'outputs' folder
print ("Export the model to model.pkl")
f = open('./outputs/model.pkl', 'wb')
pickle.dump(model, f)
f.close()
