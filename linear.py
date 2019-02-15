
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read dataset###
traindata=pd.read_csv('train.csv')
testdata=pd.read_csv('test.csv')

traindata.head()
traindata.shape
traindata.tail()
traindata.describe()
traindata.SalePrice.describe()
traindata.SalePrice.skew()


plt.hist(traindata.SalePrice,color='blue')
plt.show()

target=np.log(traindata.SalePrice)
target.skew()

plt.hist(target,color='blue')
plt.show()


numeric_features=traindata.select_dtypes(include=[np.number])
correlation=numeric_features.corr()
print(correlation)


correlation['SalePrice'].sort_values(ascending=False)
overall=traindata.pivot_table(index='OverallQual',values='SalePrice',aggfunc=np.mean)
print(overall)
overall.plot(kind='bar',color='blue')
plt.xlabel('overall quality')
plt.ylabel('sales price')
plt.xticks(rotation=0)
plt.savefig('figure1.jpg')
plt.show()



overall1=traindata.pivot_table(index='SaleCondition',values='SalePrice',aggfunc=np.mean)
print(overall1)
overall1.plot(kind='pie',subplots=True)
plt.xlabel('sales condition')
plt.ylabel('sales price')
plt.xtricks(rotation=0)
plt.savefig('figure2.jpg')
plt.show()


plt.scatter(x=traindata['GrLivArea'],y=target)
plt.xlabel('grlivarea')
plt.ylabel('sales area')
plt.show()


target1=pd.DataFrame(traindata.isnull().sum().sort_values(ascending=False))
print(target1)

Cat_features=traindata.select_dtypes(exclude=[np.number])
print(Cat_features)
Cat_features.describe()

traindata.Street.value_counts()
traindata['Encoding Street']=pd.get_dummies(traindata.Street,drop_first=True)
testdata['Encoding Street']=pd.get_dummies(testdata.Street,drop_first=True)

overall2=traindata.pivot_table(index='Encoding Street',values='SalePrice',aggfunc=np.median)
print(overall2)
overall1.plot(kind='bar',color='red')
plt.xlabel('encoding street')
plt.ylabel('sales price')
plt.xtricks(rotation=0)
plt.savefig('figure3.jpg')
plt.show()

def encode(data):return 1 if data=='normal' else 0
traindata['Encoding Sales Condition']=traindata.SaleCondition.apply(encode)

    
traindata.to_csv('newtrain.csv')
new_data=pd.read_csv('newtrain.csv')
new_data1=pd.read_csv('TRAIN_DATA.csv')

overall3=new_data.pivot_table(index='SaleType',values='SalePrice',aggfunc=np.mean)
print(overall3)

def newencode(data):
    if data=='COD':return 1
    elif data=='CWD':return 2
    elif data=='Con':return 3
    elif data=='ConLD':return 4
    elif data=='ConLI':return 5
    elif data=='ConLw':return 6
    elif data=='New':return 7
    elif data=='Oth':return 8
    elif data=='WD':return 9
new_data['Encoding Sales Type']=new_data.SaleType.apply(newencode)



#Newdata=traindata.select_dtypes(include=[np.number]).dropna().sum(traindata.isnull().sum()!=0)
#Newdata.to_csv('sathiya.csv')

#linear regression
y = np.log(new_data1.SalePrice)
X = new_data1.drop(['SalePrice'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42,test_size=.33)

from sklearn import linear_model
lr = linear_model.LinearRegression()

model = lr.fit(X_train, y_train)

print ("R^2 is: \n", model.score(X_test, y_test))

predictions = model.predict(X_test)

from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))


actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.75,
            color='b') #alpha helps to show overlapping data
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()
