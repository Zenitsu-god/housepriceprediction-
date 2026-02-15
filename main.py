import pandas as pd
from sklearn.tree import DicisionTreeRegressor

iowa_file_path =""

home_data = pd.read_csv(iowa_file_path)

home_data.describe()


y = home_data.SalesPrice
feature_columns = []

X=home_data[feature_columns]

iowa_model = DicisionTreeRegressor(random_state =1)

iowa_model.fit(X,y)

print(iowa_model.predict(X))
