# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 07:44:09 2021

@author: kkrao
"""


from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split

store = pd.HDFStore('D:/Krishna/projects/pws_drivers/data/store.h5')
df =  store['df']   # save it
store.close()
df.dropna(inplace = True)
print(df.shape)

# sub = 
X = df.drop("pws",axis = 1)
y = df['pws']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor()
regr.fit(X_train, y_train)

print(regr.score(X_test,y_test))

heights = regr.feature_importances_

order = np.argsort(heights)
ticks = X.columns
heights = heights[order]
ticks = ticks[order]

fig, ax = plt.subplots(figsize = (3,5))
ax.barh(width = regr.feature_importances_,y=range(len(X.columns)),color = "grey")
ax.set_yticks(range(len(X.columns)))
ax.set_yticklabels(X.columns)

fig, ax = plt.subplots(figsize = (3,5))
ax.barh(width = heights,y=range(len(X.columns)),color = "grey")
ax.set_yticks(range(len(X.columns)))
ax.set_yticklabels(ticks)