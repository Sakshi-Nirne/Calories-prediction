import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression

from sklearn import metrics
from sklearn.metrics import  accuracy_score,  r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBRegressor


import warnings
warnings.filterwarnings("ignore")
df_x = pd.read_csv('C:/dsbdacaloriesproject/exercise.csv')
df_y = pd.read_csv('C:/dsbdacaloriesproject/calories.csv')
df_x.head()
df_x.shape
df_y.head()
df_y.shape
x_userid = df_x['User_ID'].tolist()
y_userid = df_y['User_ID'].tolist()
print(x_userid[0:10])
print(y_userid[0:10])
missing_id = 0
avl_id = 0

for i in x_userid:
  if i in y_userid:
    avl_id += 1

  else:
    missing_id += 1


print(missing_id)
print(avl_id)
#join tebles
df_new = df_x.join(df_y, on=None, how='inner', lsuffix='User_ID', rsuffix='User_ID', sort=False)
df_new.head()
df_new.drop(['User_IDUser_ID'],axis=1,inplace=True)
df_new.head()
print(df_new.shape)
print(df_new.dtypes)
print(df_new.isnull().sum())
print(df_new)
df_new.describe().T
print(df_new['Gender'].value_counts())
plt.figure(figsize=(10,8.5))
sns.countplot(df_new['Gender'])
plt.title("Gender Value Counts",size=15)
print(plt.show())
df_new.hist(figsize=(15,15))
plt.show()
sns.distplot(df_new['Calories'])
plt.show()
#outliers detection
def outliers_check(column):

  title = str(column) + " Box Plot "
  plt.subplots(figsize=(5,5))
  sns.boxplot(data=df_new[str(i)]).set_title(title)
  plt.show()


for i in df_new[['Age',	'Height',	'Weight',	'Duration',	'Heart_Rate', 'Body_Temp',	'Calories']].columns:

    outliers_check(i)
df_new[(df_new['Height'] < 130.0) | (df_new['Height'] > 215.0)  ]
df_new[df_new['Weight'] > 130.0]
df_new[df_new['Calories'] > 300.0]
df_new[df_new['Heart_Rate'] > 125.0]
df_new[df_new['Body_Temp'] < 38.1]
label = LabelEncoder()

df_new['Gender'] = label.fit_transform(df_new['Gender'])
df_new.head(10)
plt.subplots(figsize = (14,10))
sns.heatmap(df_new.corr(),
            annot=True,fmt='.3g', vmin=-1, vmax=1, center= 0).set_title("Corelation Between Attributes")
plt.show()
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_x = df_new[['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']]

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = vif_x.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(vif_x.values, i)
                   for i in range(len(vif_x.columns))]

vif_data
pl=sns.relplot(x='Height',y='Gender',data=df_new,hue='Gender')
pl.fig.set_size_inches(7,4)
plt.show()
pl=sns.relplot(x='Height',y='Calories',data=df_new,hue='Gender',style='Gender')
pl.fig.set_size_inches(10,7)
plt.show()
pl=sns.relplot(x='Weight',y='Calories',data=df_new,hue='Gender',style='Gender')
pl.fig.set_size_inches(10,7)
plt.show()
pl=sns.relplot(x='Age',y='Calories',data=df_new,hue='Gender',style='Gender')
pl.fig.set_size_inches(10,7)
plt.show()
pl=sns.relplot(x='Duration',y='Calories',data=df_new,hue='Gender',style='Gender')
pl.fig.set_size_inches(10,7)
plt.show()
pl=sns.relplot(x='Heart_Rate',y='Calories',data=df_new,hue='Gender',style='Gender')
pl.fig.set_size_inches(10,7)
plt.show()
pl=sns.relplot(x='Body_Temp',y='Calories',data=df_new,hue='Gender',style='Gender')
pl.fig.set_size_inches(10,7)
plt.show()
sns.pairplot(df_new[['Duration','Heart_Rate', 'Body_Temp','Calories']],hue="Calories")
df_ot_hnd = pd.DataFrame(df_new[~((df_new['Height'] < 130.0) | (df_new['Height'] > 215.0))])
print(df_ot_hnd.shape)
df_ot_hnd = pd.DataFrame(df_ot_hnd[~(df_ot_hnd['Calories'] > 300.0)])
print(df_ot_hnd.shape)
df_ot_hnd = pd.DataFrame(df_ot_hnd[~(df_ot_hnd['Heart_Rate'] > 125.0)])
print(df_ot_hnd.shape)
df_ot_hnd = pd.DataFrame(df_ot_hnd[~(df_ot_hnd['Calories'] > 220.0)])
print(df_ot_hnd.shape)
print(df_ot_hnd.head())
print(df_ot_hnd.describe().T)
#data preprocessing
y = df_ot_hnd['Calories']
x = df_ot_hnd.drop(['Calories'],axis=1)
#multiple linear regression model
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
#**Multiple Linear Regression Model**
#fit model
model_1 = LinearRegression()
model_1.fit(x_train,y_train)
#training score
model1_training_score =  model_1.score(x_train,y_train)
print(f'model_1 Training Score : {model1_training_score} ')
print(f'Model 1 Coefficient : {model_1.coef_}')
print("-----------------------------------------------------")
print(f'Model 1 Coefficient : {model_1.intercept_}')
#testing score
model1_testing_score =  model_1.score(x_test,y_test)
print(f'model_1 Testing Score : {model1_testing_score} ')
y_predict = model_1.predict(x_test)

df_predictict = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict})
df_predictict.sample(10)
df_predictict['Eror'] = df_predictict['Actual'] - df_predictict['Predicted']

df_predictict.head(10)
#model evaluation
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_predict))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predict))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predict)))
r2_score(y_test,y_predict)
plt.figure(figsize=(15,8.5))

plt.scatter(y_test,y_predict, color='red')
#plt.plot(y_predict,y_test,color='black')

plt.title('Relationship Between Test Values & Prectied Values')
plt.xlabel('Actual values')
plt.ylabel('Predicted Values')
plt.show()
pl=sns.relplot(x='Actual',y='Predicted',data=df_predictict,hue='Eror')
pl.fig.set_size_inches(10,7)
plt.show()
#XG Boost Regressior
#XGBoost makes predictions by combining the predictions of
#multiple decision trees. The decision trees are trained
# on the input data and the corresponding output values,
# using a gradient boosting approach to iteratively improve
# the accuracy of the model.
model_2 = XGBRegressor()
print(model_2.fit(x_train,y_train))
#training score
model_2_training_score = model_2.score(x_train,y_train)
print(f'model_2 Training Score : {model_2_training_score}')
y_predict_2 = model_2.predict(x_test)

df_predictict_2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict_2})
print(df_predictict_2.sample(10))
df_predictict_2['Eror'] = df_predictict_2['Actual'] - df_predictict_2['Predicted']
print(df_predictict_2.sample(10))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_predict_2))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predict_2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predict_2)))
print(r2_score(y_test,y_predict_2))
pl=sns.relplot(x='Actual',y='Predicted',data=df_predictict_2,hue='Eror')
pl.fig.set_size_inches(10,7)
plt.show()
input_data = ['0',	'27',	'154.0',	'58.0',	'10.0',	'81.0',	'39.8']

input_array = np.asarray(input_data)
input_array_reshape = input_array.reshape(1,-1)

#StandardScaler removes the mean and scales each feature/variable to unit variance
sc_2 = StandardScaler()
cal_x = sc.transform(input_array_reshape)

Caloiries_pred =  model_2.predict(cal_x)
print(Caloiries_pred)
import pickle
#Pickle allows for flexibility when deserializing objects .
# You can easily save different variables into a Pickle file and
# load them back in a different Python session, recovering your
# data exactly the way it was without having to edit your code

with open('Calories_model', 'wb') as f:
    pickle.dump(model_2, f)
