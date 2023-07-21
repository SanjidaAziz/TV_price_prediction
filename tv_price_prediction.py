import pandas as pd
import numpy as np
import sklearn
print('sklearn: %s' % sklearn.__version__)
from sklearn import metrics

from sklearn.model_selection import train_test_split
ds = pd.read_csv('Dataset.csv')
X = ds.drop(['Price'], axis=1)
Y = ds['Price']

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

#Brand
brands = pd.get_dummies(X['Brand'])
X = X.drop(['Brand'], axis=1)
X = pd.concat([X, brands], axis=1)
#print(brands.unique())


#Resolution
resolution = pd.get_dummies(X['Resolution'])
X = X.drop(['Resolution'], axis=1)
X = pd.concat([X, resolution], axis=1)


#DeviceType
deviceType = pd.get_dummies(X['DeviceType'])
X = X.drop(['DeviceType'], axis=1)
X = pd.concat([X, deviceType], axis=1)


#SpeakerSystem
speakerSystem = pd.get_dummies(X['SpeakerSystem'])
X = X.drop(['SpeakerSystem'], axis=1)
X = pd.concat([X, speakerSystem], axis=1)


#Resolutionupscaler
resolutionupscaler = pd.get_dummies(X['Resolutionupscaler'])
X = X.drop(['Resolutionupscaler'], axis=1)
X = pd.concat([X, resolutionupscaler], axis=1)


x_train, xtest, y_train, ytest = train_test_split(X, Y, test_size=.25, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

prediction = regressor.predict(xtest)
linear_error = "{:.2f}".format(metrics.mean_absolute_percentage_error(ytest, prediction) * 100)
print()
print('Performance for Multiple Linear Regression:')
print('-------------------------------------------')
print('Mean absolute percentage error: ', linear_error, ' %')
print('Mean Absolute Error:', 	metrics.mean_absolute_error(ytest, prediction))
print('Mean Squared Error:', metrics.mean_squared_error(ytest, prediction))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(ytest, prediction)))
print('R2 Score:', metrics.r2_score(ytest, prediction))


from sklearn import tree

regressor = tree.DecisionTreeRegressor()
regressor.fit(x_train, y_train)

prediction = regressor.predict(xtest)
decision_tree_error = "{:.2f}".format(metrics.mean_absolute_percentage_error(ytest, prediction) * 100)
print()
print('Performance for Decision Tree Regression:')
print('-------------------------------------------')
print('Mean absolute percentage error: ', decision_tree_error, ' %')
print('Mean Absolute Error:', 	metrics.mean_absolute_error(ytest, prediction))
print('Mean Squared Error:', metrics.mean_squared_error(ytest, prediction))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(ytest, prediction)))
print('R2 Score:', metrics.r2_score(ytest, prediction))

from sklearn.neighbors import KNeighborsRegressor

regressor = KNeighborsRegressor(n_neighbors=18)
regressor.fit(x_train, y_train)

prediction = regressor.predict(xtest)
knn_error = "{:.2f}".format(metrics.mean_absolute_percentage_error(ytest, prediction) * 100)
print()
print('Performance for K Nearest Neighbor(KNN):')
print('-------------------------------------------')
print('Mean absolute percentage error: ', knn_error, ' %')
print('Mean Absolute Error:', 	metrics.mean_absolute_error(ytest, prediction))
print('Mean Squared Error:', metrics.mean_squared_error(ytest, prediction))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(ytest, prediction)))
print('R2 Score:', metrics.r2_score(ytest, prediction))

from sklearn.naive_bayes import GaussianNB

regressor = GaussianNB()
regressor.fit(x_train, y_train)

prediction = regressor.predict(xtest)
gaussian_error = "{:.2f}".format(metrics.mean_absolute_percentage_error(ytest, prediction) * 100)
print()
print('Performance for Gaussian Naive Bayes:')
print('-------------------------------------------')
print('Mean absolute percentage error: ', gaussian_error, ' %')
print('Mean Absolute Error:', 	metrics.mean_absolute_error(ytest, prediction))
print('Mean Squared Error:', metrics.mean_squared_error(ytest, prediction))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(ytest, prediction)))
print('R2 Score:', metrics.r2_score(ytest, prediction))

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=200)
regressor.fit(x_train, y_train)

prediction = regressor.predict(xtest)
random_forest_error = "{:.2f}".format(metrics.mean_absolute_percentage_error(ytest, prediction) * 100)
print()
print('Performance for Random Forest:')
print('-------------------------------------------')
print('Mean absolute percentage error: ', random_forest_error, ' %')
print('Mean Absolute Error:', 	metrics.mean_absolute_error(ytest, prediction))
print('Mean Squared Error:', metrics.mean_squared_error(ytest, prediction))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(ytest, prediction)))
print('R2 Score:', metrics.r2_score(ytest, prediction))

Z = X.head(1)

if random_forest_error <= decision_tree_error and random_forest_error <= gaussian_error and random_forest_error <= knn_error and random_forest_error <= linear_error:
    regressor = RandomForestRegressor()
else:
    if decision_tree_error <= gaussian_error and decision_tree_error <= knn_error and decision_tree_error <= linear_error:
        regressor = tree.DecisionTreeRegressor()
    else:
        if gaussian_error <= knn_error and gaussian_error <= linear_error:
            regressor = GaussianNB()
        else:
            if knn_error <= linear_error:
                regressor = KNeighborsRegressor()
            else:
                regressor = LinearRegression()

print("\n\nBest model based on performance: ", regressor)


#Taking User Input
print()
print("Brands:\n--------------\nJamuna Konka LG Marcel Minister MyOne Samsung Singer Sony Vision Walton     Xiaomi Mi")
input_brand = input("Enter Brand: ")
input_ScreenSize = int(input("Enter Screen Size: "))
print("Resolution:\n---------------------\n1280x720 1366x768 1920x1080 3840x2160 3841x2160 4320x2160 720x576")
input_resolution = input("Enter Resolution: ")
print("Device Type::\n--------------\nCRT FHD LCD LED OLED QLED UHD")
input_deviceType = input("Enter Device Type: ")
input_powerSupply = int(input("Enter Power Supply: "))
input_audioOutput = int(input("Enter Audio Output: "))
print("Speaker System:\n--------------------------\n2.0Channel 2.2Channel 4.0Channel 4.2Channel Integrated")
input_speakerSystem = input("Enter Speaker System: ")
input_hdmi = int(input("Enter HDMI: "))
input_usb = int(input("Enter USB: "))
input_smartTV = int(input("Enter Smart TV: "))
print("Resolution Upscaler:\n-------------------------\n4k 4KHDR 4KUHD FHD HD SD UHD")
input_resolutionupscaler = input("Enter Resolution upscaler: ")


Z.loc[0, input_brand] = 1
Z.loc[0, 'ScreenSize'] = input_ScreenSize
Z.loc[0, input_resolution] = 1
Z.loc[0, input_deviceType] = 1
Z.loc[0, 'PowerSupply'] = input_powerSupply
Z.loc[0, 'AudioOutput'] = input_audioOutput
Z.loc[0, input_speakerSystem] = 1
Z.loc[0, 'HDMI'] = input_hdmi
Z.loc[0, 'USB'] = input_usb
Z.loc[0, 'Smart TV'] = input_smartTV
Z.loc[0, input_resolutionupscaler] = 1


regressor.fit(x_train, y_train)
predictedPrice = regressor.predict(Z)
print("Predicted Price : ", predictedPrice, " Tk")
