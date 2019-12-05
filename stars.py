import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from scipy.stats import mode
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from statsmodels.formula.api import ols
%matplotlib inline
​
stars = pd.read_csv('6 class csv.csv')
stars.head()
Temperature (K)	Luminosity(L/Lo)	Radius(R/Ro)	Absolute magnitude(Mv)	Star type	Star color	Spectral Class
0	3068	0.002400	0.1700	16.12	0	Red	M
1	3042	0.000500	0.1542	16.60	0	Red	M
2	2600	0.000300	0.1020	18.70	0	Red	M
3	2800	0.000200	0.1600	16.65	0	Red	M
4	1939	0.000138	0.1030	20.06	0	Red	M
#We will start by looking at the data to see where everything stands and to check for null values.
stars.describe()

#Looks like some stars have extremely low and extremely high luminosity, lets dig a little deeper
sns.boxplot(stars['Luminosity(L/Lo)'])

#Looks like those could easily be outliers so we should be careful about how we handle the data.

plt.figure(figsize=(10,5))
sns.distplot(stars['Temperature (K)'])


sns.distplot(stars['Luminosity(L/Lo)'])

sns.distplot(stars['Radius(R/Ro)'])

sns.distplot(stars['Absolute magnitude(Mv)'])

sns.countplot(y=stars['Star color'])

sns.countplot(y=stars['Color'])

sns.countplot(y=stars['Star color'])

sns.countplot(stars['Star type'])

sns.countplot(stars['Spectral Class'])

#Now we csn see how everything is distributed. 
#Depending on how well the model does we might make a separate one without the magnitude outliers.
#Before that we need to convert the spectral classes and Color into numeric values. 
#Color had a lot of problems so we fixed those right up with a nice disctionary replace statement

stars['Spectral Class'].replace({"A":1,"B":2,"F":3,"G":4,"K":5,"M":6,"O":7},inplace=True)
stars['Star Color'] = stars['Star color'].replace({"Red":1,"Blue White":2,"White":3,"Yellowish White":4,
                                                   "Blue white":2,"Pale yellow orange":5,"Blue":6,"Blue-white":2,
                                                   "Whitish":2,"yellow-white":4,"Orange":5,"White-Yellow":4,"white":3,
                                                   "blue":6,"Blue ":6,"yellowish":4,"Yellowish":4,"Orange-Red":1,"Blue white ":2,"Blue-White":2})
​
​
​
​
​
stars.drop('Star color',axis=1,inplace=True)
#Next lets rename columns for simplicity

stars.columns = ['Temp','Lum','Radius','AbsMag','Type','Class','Color']
stars.head()

#Now we will look at collinearity and significance with a pair plot and an anova

sns.pairplot(stars,hue='Type1')

#It doesn't look like there are many correlations except for maybe magnitude and temperate.
#Let's check the correlation heatmap before continuing
#It definitely looks like there are clusters in the data so using KNN or K means might be a good approach
#Before anything else we will transform the data to make it normalized
​
stars['Type1'] = stars['Type']
stars.head()

stars.drop('Type',axis=1,inplace=True)
scaler = StandardScaler()
scaler.fit(stars.drop('Type1',axis=1))

StandardScaler(copy=True, with_mean=True, with_std=True)
scaled_features = scaler.transform(stars.drop('Type1',axis=1))

scaler = StandardScaler()
scaler.fit(stars.drop('Type1',axis=1))
scaled_features = scaler.transform(stars.drop('Type1',axis=1))
stars_feat = pd.DataFrame(scaled_features, columns=stars.columns[:-1])

stars_feat.head()

stars_feat.describe()

#This will be for final graphing

final = stars_feat
final['Type'] = stars['Type1']
final.head()

ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
);

#Clearly there aren't any variables that are highly correlated, so we don't have to do any variable alterations.

results = ols('Type1 ~ Temp + Lum + Radius + AbsMag + Class + Color', data=stars).fit()
aov_table = sm.stats.anova_lm(results, typ=2)
aov_table

#According to this every variable is significantly correlated with the response variable.
#Now we will split the data into an estimation and validation set and try out 

y = stars['Type1']
X_train, X_test, y_train, y_test = train_test_split(stars_feat, y, test_size=0.25, random_state=1)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=None, n_neighbors=1, p=2,
           weights='uniform')
           
pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

#Obviously these results seem too good to be true, so we will dig deeper to understand why such a low k value works so well.
#Generally you would think I am overfitting a ton but maybe not so we will check other k values with this cool plot.

error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    
plt.figure(figsize=(12,6))
plt.plot(range(1,40),error_rate, color='blue',linestyle='dashed',marker='o',
        markerfacecolor='red',markersize=10)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
Text(0, 0.5, 'Error Rate')

#Very interesting how the error rate increases from the beginning at 1 then levels out after 15 at .6, which is horrible.
#1 doesn't make sense intuitively but this is on the test set. 
#But after resetting the train and test on numerous random scales it seemed to hold
#Let's graph it to see if it looks correct

final.head()

sns.scatterplot(x='Temp',y='Lum',data=final,hue='Type')

sns.scatterplot(x='AbsMag',y='Color',data=final,hue='Type')

sns.scatterplot(x='Class',y='Radius',data=final,hue='Type')

fig = px.scatter_3d(final, x='Color', y='Radius', z='Class',
              color='Type')
fig.show()

fig = px.scatter_3d(final, x='Temp', y='Lum', z='AbsMag',
              color='Type')
fig.show()

#Looking at the plots it does look like stars align with whatever is next to them and there are very clear 
#clusters so knn would be a great fit for this model.
#Let's look at some descriptions for each category of star to see how much variance there is

sns.pairplot(final[final['Type']==1].drop('Type',axis=1))

sns.pairplot(final[final['Type']==2].drop('Type',axis=1))

sns.pairplot(final[final['Type']==3].drop('Type',axis=1))

sns.pairplot(final[final['Type']==4].drop('Type',axis=1))

sns.pairplot(final[final['Type']==5].drop('Type',axis=1))

