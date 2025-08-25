import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
path = os.getcwd()

#Load the Churn_Modeling data set
df = pd.read_csv('Dataset/HousePrices.csv')

plt.style.use('_mpl-gallery')
x = df['condition']
y = df['price']

#plot
fig, ax = plt.subplots()
ax.plot(x, y/1E6, 'X', markeredgewidth=0.5)
ax.set_ylabel("Price in Millions")
ax.set_xlabel("House Condition 1-5")

from sklearn.preprocessing import OneHotEncoder
dropper = ['date', 'country', 'street', 'yr_built', 'sqft_above']
df_copy = df.drop(dropper, axis=1)
ohe = OneHotEncoder(drop=None, sparse_output=False)
city_ohe = ohe.fit_transform(df_copy[['city']])

city_df = pd.DataFrame(city_ohe, columns=ohe.get_feature_names_out(['city']), index=df_copy.index)

df_city_ohe = pd.concat([df_copy.drop('city', axis=1), city_df], axis=1)

statezip_ohe = ohe.fit_transform(df_city_ohe[['statezip']])

statezip_df = pd.DataFrame(statezip_ohe, columns=ohe.get_feature_names_out(['statezip']), index=df_city_ohe.index)

df_statezip_ohe = pd.concat([df_city_ohe.drop('statezip', axis=1), statezip_df], axis=1)


high = 0.99
low = 0.01

df_filtered = df_statezip_ohe[(df_statezip_ohe['price'] > df_statezip_ohe['price'].quantile(low)) & (df_statezip_ohe['price'] < df_statezip_ohe['price'].quantile(high))]


from sklearn.model_selection import train_test_split
Y = df_filtered['price']
X = df_filtered.drop('price', axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
model = LinearRegression()
model.fit(X_train, Y_train)

y_predicted = model.predict(X_test)

mse = mean_squared_error(Y_test, y_predicted)
r2 = r2_score(Y_test, y_predicted)

print(f"Residual error: {mse:,.2f}") 
print(f"Percentage of certainty: {r2:,.2f}%")
plt.show()