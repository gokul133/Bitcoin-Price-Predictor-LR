import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.preprocessing import minmax_scaling
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


df = pd.read_csv("/Users/gokulk/Downloads/BTC-USD.csv")
df1 = df.copy(deep=True)
df1 = df1.drop(['Date', 'High', 'Open', 'Low', 'Volume', 'Adj Close'], axis=1)


prediction_days = int(input("Please enter how many days you want to predict the data for Bitcoin close price : "))

# df1['Open_Prediction'] = df[['Open']].shift(-prediction_days)
df1['Close_Prediction'] = df1['Close'].shift(-prediction_days)
# X = np.array(df1.drop(['Open_Prediction'], axis=1))
X = np.array(df1.drop(['Close_Prediction'], axis=1))
# X = X[:len(df1)-prediction_days]
X = X[:len(df1)-prediction_days]

# print(Y)

Y = np.array(df1['Close_Prediction'])
Y = Y[:-prediction_days]


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=42)

x_train1 = x_train
x_test1 = x_test


x_train = x_train.flatten()
x_test = x_test.flatten()

x_train3 = x_train
x_test3 = x_test
y_train3 = y_train

prediction_days_array = np.array(df1.drop(['Close_Prediction'], axis=1))
prediction_days_array = prediction_days_array[-prediction_days:]



print("********Using Linear Regression**********")



x_train_mean = 0
for i in x_train:
    x_train_mean = x_train_mean+i
x_train_mean = x_train_mean/len(x_train)

y_train_mean = 0
for i in y_train:
    y_train_mean = y_train_mean+i
y_train_mean = y_train_mean/len(y_train)

sum1=0
sum2=0
for i in range(0, len(x_train)):
    sum1 = sum1+(x_train[i]-x_train_mean)*(y_train[i]-y_train_mean)
    sum2 = sum2+(x_train[i]-x_train_mean)*(x_train[i]-x_train_mean)
m = sum1/sum2
c = y_train_mean-(m*x_train_mean)

x1=np.array(x_train)
y1=np.array(y_train)
plt.scatter(x1, y1, color="m",marker="o", s=30)

# predicted response vector
y_pred = m*x1 + c

# plotting the regression line
plt.title(label='Train data Linear Regression Plot')
plt.plot(x1, y_pred, color="g")

# putting labels
plt.xlabel('x')
plt.ylabel('y')

# function to show plot
plt.show()


print("\n\n----------------------Linear Regression using User Defined function------------------------")
s1=0
s2=0
mse1=0
for i in range(0, len(y_pred)):
    s1 = s1+(y_pred[i]-y_train_mean)*(y_pred[i]-y_train_mean)
    s2 = s2+(y1[i]-y_train_mean)*(y1[i]-y_train_mean)
    mse1 = mse1+(y1[i]-y_pred[i])*(y1[i]-y_pred[i])

mse1 = mse1/len(y_pred)

r2 = s1/s2
print("R square value =", r2)
#
#
# print("The Mean Squared error is : ", mse1)

m = sum1/sum2
c = y_train_mean-(m*x_train_mean)



x1=np.array(x_test)
y1=np.array(y_test)
plt.scatter(x1, y1, color="m",marker="o", s=30)

# predicted response vector
y_pred = m*x1 + c

# plotting the regression line
plt.title(label='Test data Linear Regression Plot')
plt.plot(x1, y_pred, color="g")

# putting labels
plt.xlabel('x')
plt.ylabel('y')

# function to show plot
plt.show()

mntst = 0
for i in y1:
    mntst = mntst+i
mntst = mntst/len(y1)


# s11 = 0
# s21 = 0
# for i in range(0, len(y_pred)):
#     s11 = s11+(y_pred[i])*(y_pred[i])
#     s21 = s21+(y1[i])*(y1[i])
#
# r21 = s11/s21
# print("R square value for test data predicted values =",r21)


print("The predicted values for the next", prediction_days, "Days are : ")
for i in prediction_days_array:
    print(m*i+c)


#with Predifined function

print("\n\n----------------------Linear Regression using Pre-Defined Function---------------------------")
reg = linear_model.LinearRegression()
reg.fit(x_train1, y_train)


reg_confidence = reg.score(x_train1, y_train)
print("R square value = ", reg_confidence)

#
#
# print("The Mean Squared error is : ", mean_squared_error(y_train, reg.predict(x_train1)))

reg_predict = reg.predict(x_test1)

reg_prediction = reg.predict(prediction_days_array)

#predictions for the next N days
print("The predicted values for the next", prediction_days, "Days are : ")
print(reg_prediction)


print("\n\n**********Using KNN user defined**********")

x_train4 = np.array(x_train3)
y_train4 = np.array(y_train3)

y_train4_mean = 0
for i in y_train4:
    y_train4_mean = y_train4_mean+i
y_train4_mean = y_train4_mean/len(y_train4)

y_pred4 = []
max = max(x_train4)
for i in range(0,len(x_train4)):
    lst = []
    for j in range(0,len(x_train4)):
        val = x_train4[i]-x_train4[j]
        if val<0:
            val=val*(-1)
        lst.append(val)
    pred4 = 0
    for i in range(0,5):
        ind = lst.index(min(lst))
        lst[ind] = max
        pred4 = pred4+y_train4[ind]
    y_pred4.append(pred4/5)

s4=0
s5=0
for i in range(0, len(y_pred4)):
    s4 = s4+(y_pred4[i]-y_train4_mean)*(y_pred4[i]-y_train4_mean)
    s5 = s5+(y_train4[i]-y_train4_mean)*(y_train4[i]-y_train4_mean)


r24 = s4/s5
print("R square value = ", r24)

y_pred4 = []
max = prediction_days_array.max()
for i in range(0,len(prediction_days_array)):
    lst = []
    for j in range(0,len(x_train4)):
        val = prediction_days_array[i]-x_train4[j]
        if val<0:
            val=val*(-1)
        lst.append(val)
    pred4 = 0
    for i in range(0,5):
        ind = lst.index(min(lst))
        lst[ind] = max
        pred4 = pred4+y_train4[ind]
    y_pred4.append(pred4/5)
print("\nThe predicted values for the next", prediction_days, "Days are : ")
print(y_pred4)



print("\n\n**********Using KNN pre defined**********")
knnreg = KNeighborsRegressor(n_neighbors=5)
knnreg.fit(x_train1, y_train)


knnreg_confidence = knnreg.score(x_train1, y_train)
print("R square value = ", knnreg_confidence)

knnreg_predict = knnreg.predict(x_test1)

#
# print("The predicted values for the next", prediction_days, "Days are : ")
# print(knnreg_predict)


knnreg_prediction = knnreg.predict(prediction_days_array)

#predictions for the next N days
print("The predicted values for the next", prediction_days, "Days are : ")
print(knnreg_prediction)
#
# print()

#print the actual price of bitcoin for last 30 days
# print(df.tail(prediction_days))
# print(knnreg_predict)




print("\n\n---------------Linear Regression using User Defined function with Open and Close---------------")

x_train = np.array(df["Open"])
y_train = np.array(df["Close"])

x_train_mean = 0
for i in x_train:
    x_train_mean = x_train_mean+i
x_train_mean = x_train_mean/len(x_train)

y_train_mean = 0
for i in y_train:
    y_train_mean = y_train_mean+i
y_train_mean = y_train_mean/len(y_train)

sum1=0
sum2=0
for i in range(0, len(x_train)):
    sum1 = sum1+(x_train[i]-x_train_mean)*(y_train[i]-y_train_mean)
    sum2 = sum2+(x_train[i]-x_train_mean)*(x_train[i]-x_train_mean)
m = sum1/sum2
c = y_train_mean-(m*x_train_mean)

x1=np.array(x_train)
y1=np.array(y_train)
plt.scatter(x1, y1, color="m",marker="o", s=30)

# predicted response vector
y_pred = m*x1 + c

# plotting the regression line
plt.title(label='Train data Linear Regression Plot 2')
plt.plot(x1, y_pred, color="g")

# putting labels
plt.xlabel('x')
plt.ylabel('y')

# function to show plot
plt.show()


s1=0
s2=0
mse1=0
for i in range(0, len(y_pred)):
    s1 = s1+(y_pred[i]-y_train_mean)*(y_pred[i]-y_train_mean)
    s2 = s2+(y1[i]-y_train_mean)*(y1[i]-y_train_mean)
    mse1 = mse1+(y1[i]-y_pred[i])*(y1[i]-y_pred[i])

mse1 = mse1/len(y_pred)

r2 = s1/s2
print("R square value =", r2)
#
#
# print("The Mean Squared error is : ", mse1)

x = float(input("Enter open price of predicting day:"))
print("The predicted value of close price is: ")
print(m*x+c)