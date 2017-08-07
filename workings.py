
# coding: utf-8

# # Quick data modification if we just want to jump to the Models with the appropriate data structure

# In[37]:

#Note: Attribution for code samples/credit to be added as we go.
# many based on code sample for ployly, kaggle, sklearn
#attribution to be added as I refind them

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

#Try and figure out where we actually are in the filesystem
import os
os.getcwd()

#Read in our Data
diamonds = pd.read_csv("diamonds.csv") #load the dataset

# Drop zero values from the dataset     
diamonds = diamonds[(diamonds.x != 0)]
diamonds =  diamonds[(diamonds.y !=0)]
diamonds = diamonds[(diamonds.z != 0)]

#Drop Outliers
diamonds.drop(diamonds[diamonds.y == 58.9].index, inplace=True)
diamonds.drop(diamonds[diamonds.y == 31.8].index, inplace=True)
diamonds.drop(diamonds[diamonds.z == 31.8].index, inplace=True)

#Drop the Unnamed Column
diamonds.drop(diamonds.columns[[0]], axis=1,inplace=True)  # diamonds.columns is zero-based pd.Index 

##Another candidate for removal
diamonds.drop(diamonds[diamonds.table == 95.0].index, inplace=True)

#Add log10 of price
#diamonds['lprice']=np.log10(diamonds.price)

#Add natural log of price
diamonds['lprice']=np.log(diamonds.price)

## Add a flag for Ideal cut only - Ideal= True, all others = False
diamonds['ideal_flag'] = diamonds['cut'].apply(lambda x: 'True' if x == 'Ideal' else 'False')






# In[ ]:

### Train/Test split


# In[38]:




#Fancy train/test split

from sklearn.cross_validation import train_test_split

#X_all = diamonds.drop(['ideal_flag','cut'], axis=1)
X_all = diamonds.drop(['ideal_flag'], axis=1)
Y_all = diamonds['ideal_flag']

num_test = 0.30
#train_X, test_X, train_y, test_y = train_test_split(X_all, Y_all, test_size=num_test, random_state=23)
diamonds_train, diamonds_test = train_test_split(diamonds,test_size=num_test,random_state=23)


#Some Final Encoding based on #https://www.kaggle.com/enerrio/scikit-learn-ml-from-start-to-finish


from sklearn import preprocessing
def encode_features(df_train, df_test):
    features = ['cut','clarity','color']
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test
    
diamonds_train, diamonds_test = encode_features(diamonds_train, diamonds_test)

#diamonds_train.head()

#Set up our training and test data
train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
train_y=diamonds_train[['lprice']] # output of our training data
test_X =diamonds_test[['carat','cut','clarity','color','x','y','z','depth','table']] # taking test data features
test_y=diamonds_test[['lprice']]  #output value of test data



# In[39]:

diamonds_train.describe()


# In[40]:

diamonds_test.describe()


# In[34]:

diamonds.describe()


# In[35]:

np.log(18823)


# In[36]:

np.exp(9.8428348053241788)


# ## Data exploration of the original dataset and all that follows

# In[3]:

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
#import math as mat_lib


#points of interest


#price price in US dollars (\$326--\$18,823)

#carat weight of the diamond (0.2--5.01)

#cut quality of the cut (Fair, Good, Very Good, Premium, Ideal)

#color diamond colour, from J (worst) to D (best) (reverse alphabetical order) J I H G F E D

#clarity a measurement of how clear the diamond is 
#(I1 Included (worst), SI2 (Slightly Included grade 2), SI1, VS2 (very slightly included grade 2), VS1, 
#VVS2 (very very slightly included grade 2), VVS1, IF (Internally Flawless)(best))

#x length in mm (0--10.74)

#y width in mm (0--58.9)

#z depth in mm (0--31.8)

#depth total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)

#table width of top of diamond relative to widest point (43--95)


#carat - weight of diamond
#cut - shape diamond is cut
#color - white, blue etc
#clarity - inclusions etc
#depth has value ranging for 0 to :99999999
#table sees redundant
#price - What is sells for
#x dimension
#y dimension
#z dimension

#Try and figure out where we actually are in the filesystem
import os
os.getcwd()

#Read in our Data
diamonds = pd.read_csv("diamonds.csv") #load the dataset


# #Take an initial look at the data
# diamonds.head(5)
# 

# In[9]:

diamonds.info()


# In[10]:

diamonds.describe()


# In[6]:

pd.__version__


# In[8]:

diamonds.describe(include = 'all')


# In[7]:

diamonds.describe(include = ['O'])


# In[7]:

diamonds.hist(edgecolor='black', linewidth=1.2)

plt.show()
# Release memory.
plt.clf()
plt.close()


# In[3]:

sns.pairplot(diamonds,x_vars=['carat','depth','table','x','y','z'],y_vars='price')

plt.show()
# Release memory.
plt.clf()
plt.close()


# In[7]:

plt.figure(figsize=(12,10))
plt.subplot(3,3,1)
sns.boxplot(x="cut", y="price",data=diamonds,order=['Fair','Good','Very Good','Premium','Ideal'])
plt.subplot(3,3,2)
sns.boxplot(x='cut',y='carat',data=diamonds,order=['Fair','Good','Very Good','Premium','Ideal'])
plt.subplot(3,3,3)
sns.violinplot(x='cut',y='price',data=diamonds,order=['Fair','Good','Very Good','Premium','Ideal'])
plt.subplot(3,3,4)
sns.violinplot(x='cut',y='carat',data=diamonds,order=['Fair','Good','Very Good','Premium','Ideal'])
plt.subplot(3,3,5)
sns.violinplot(x='table',y='price',data=diamonds)
plt.subplot(3,3,6)
sns.violinplot(x='depth',y='price',data=diamonds)
sns.boxplot(x='color',y='price',data=diamonds,order=['J','I','H','G','F','E', 'D'])
plt.subplot(3,3,7)
sns.violinplot(x='color',y='price',data=diamonds,order=['J','I','H','G','F','E', 'D'])
plt.subplot(3,3,8)
sns.violinplot(x='color',y='carat',data=diamonds,order=['J','I','H','G','F','E', 'D'])

plt.show()
# Release memory.
plt.clf()
plt.close()


# ### Prive versus cut, color, clarity

# In[4]:

plt.figure(figsize=(12,10))
plt.subplot(3,3,1)
sns.violinplot(x="cut", y="price",data=diamonds,order=['Fair','Good','Very Good','Premium','Ideal'])
plt.subplot(3,3,2)
sns.violinplot(x='color',y='price',data=diamonds,order=['J','I','H','G','F','E', 'D'])
plt.subplot(3,3,3)
sns.violinplot(x='clarity',y='price',data=diamonds,order=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'])
plt.show()
# Release memory.
plt.clf()
plt.close()


# In[42]:

sns.countplot(x="price", data=diamonds)  #data=diamonds[diamonds.price<350]
plt.show()
# Release memory.
plt.clf()
plt.close()


# In[5]:

diamonds.price[diamonds.price<500].count


# In[4]:

plt.figure(figsize=(12,10))
plt.subplot(3,3,1)
sns.distplot(diamonds.price, kde=False, color="b") #Bins based on scale so different shape if subset
plt.subplot(3,3,2)
sns.distplot(diamonds.table, kde=False, color="b")
plt.subplot(3,3,3)
sns.distplot(diamonds.carat, kde=False, color="b")
plt.show()
# Release memory.
plt.clf()
plt.close()


# In[9]:

diamonds.describe()


# In[ ]:




# In[8]:

#plt.figure(figsize=(12,10))
#plt.subplot(2,2,1)
#sns.violinplot(x='price',y='x',data=diamonds)
#plt.subplot(2,2,2)
#sns.violinplot(x='price',y='y',data=diamonds)
#plt.subplot(2,2,3)
#sns.violinplot(x='price',y='z',data=diamonds)
#plt.subplot(2,2,4)
#sns.violinplot(x='price',y='z',data=diamonds)
#plt.show()
# Release memory.
#plt.clf()
#plt.close()


# In[36]:

#sns.boxplot([diamonds.price])
#plt.show()


# ### Review the values for x, y, z 

# In[ ]:

#### Lets see what's going on with x, y, z


# In[7]:

plt.figure(figsize=(12,10))
sns.pairplot(diamonds,x_vars=['x','y','z'],y_vars='price')
plt.show()


# In[7]:

diamonds[diamonds.x==0]
diamonds[diamonds.y==0]
diamonds[diamonds.z==0]


# In[3]:

diamonds_temp=diamonds[(diamonds.x ==0) | (diamonds.y ==0) | (diamonds.z == 0)]
#diamonds[diamonds['z']<1]


# In[13]:

diamonds[(diamonds.x ==0) | (diamonds.y ==0) | (diamonds.z == 0)]


# In[41]:

diamonds_temp.count() # So we have 20 or so rows with zero values


# In[62]:

diamonds.describe()


# In[ ]:

#We can see that x y and z have min values of zero - does that make sense - omly if we have a planar surface


# ### Drop the Zero values

# In[2]:

#diamonds = diamonds[(diamonds.x != 0)| (diamonds.y !=0) | (diamonds.z != 0)]
# So lets drop them from the dataset     
diamonds = diamonds[(diamonds.x != 0)]
diamonds =  diamonds[(diamonds.y !=0)]
diamonds = diamonds[(diamonds.z != 0)]


# In[9]:

#So let's replot these data and see what else we can see
sns.pairplot(diamonds,x_vars=['carat','depth','table','x','y','z'],y_vars='price')
plt.show()
# Release memory.
plt.clf()
plt.close()


# In[ ]:

#We can see that there a few outliers - out to the righthand sidde - better take care of them


# In[10]:

#So what are the max values for these variables
print(' Max Value for x:',max(diamonds.x))
print(' Max Value for y:',max(diamonds.y))
print(' Max Value for z:',max(diamonds.z))


# In[3]:

# x looks OK, but well, y and z need attention


# In[13]:

diamonds[diamonds.y>55]


# In[14]:

#Just check we're looking at the right row

diamonds.loc[24067]


# ### Drop the Outliers from x, y, z and table

# #### Drop values from y 

# In[3]:

#df.drop(df[df.score < 50].index, inplace=True)
#df = df.drop(df[(df.score < 50) & (df.score > 20)].index)

diamonds.drop(diamonds[diamonds.y == 58.9].index, inplace=True)


# In[12]:

#Just check we got out target
print (diamonds.describe())
print

print(' Max Value for x:',max(diamonds.x))
print(' Max Value for y:',max(diamonds.y))
print(' Max Value for z:',max(diamonds.z))


# In[13]:

#So let's replot these data and see what else we can see
sns.pairplot(diamonds,x_vars=['carat','depth','table','x','y','z'],y_vars='price')
plt.show()
# Release memory.
plt.clf()
plt.close()


# #### Another case in y

# In[17]:

diamonds[diamonds.y>31]


# In[4]:

#Another candidate for removal

diamonds.drop(diamonds[diamonds.y == 31.8].index, inplace=True)


# In[9]:

#Just check we got out target
print (diamonds.describe())
print

print(' Max Value for x:',max(diamonds.x))
print(' Max Value for y:',max(diamonds.y))
print(' Max Value for z:',max(diamonds.z))


# #### Drop value in z
# 

# In[5]:

diamonds.drop(diamonds[diamonds.z == 31.8].index, inplace=True)


# In[20]:

#Just check we got out target
print (diamonds.describe())


# In[7]:

#Reindex the dataframe
diamonds.reset_index(drop=True, inplace=True)


# In[9]:

diamonds.describe()


# In[18]:

#So let's replot these data and see what else we can see
sns.pairplot(diamonds,x_vars=['carat','depth','table','x','y','z'],y_vars='price')
plt.show()
plt.clf()
plt.close()


# #### Drop the Unnamed Column

# In[6]:

#Drop the Unnamed Column
diamonds.drop(diamonds.columns[[0]], axis=1,inplace=True)  # diamonds.columns is zero-based pd.Index 
#drop it


# In[15]:

#Lets look at that Table value out on its own
print(' Max Value for z:',max(diamonds.table))


# #### Check the Outlier in table

# In[34]:

#we can see from the quantiles that this value is a bit extreme
np.percentile(diamonds.table,(90,99,99.99,99.999))


# #### Drop the value from table

# In[7]:

##Another candidate for removal

diamonds.drop(diamonds[diamonds.table == 95.0].index, inplace=True)


# In[10]:

diamonds.describe()


# In[8]:

#Reindex
diamonds.index = range(1,len(diamonds) + 1)


# In[80]:

#df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'}, inplace=True)
#diamonds.rename(columns={'lprice': 'log_price'}, inplace=True)


# In[ ]:

#### add new column based on Log of the price 


# In[29]:

plt.figure(figsize=(12,10))
plt.subplot(3,2,1)
sns.distplot(diamonds.price, kde=False, color="b") #Bins based on scale so different shape if subset
plt.subplot(3,2,2)
sns.distplot(np.log10(diamonds.price), kde=False, color="b",axlabel='Log10 of Price')

plt.show()
# Release memory.
plt.clf()
plt.close()


# In[41]:

plt.figure(figsize=(12,10))
plt.subplot(3,2,1)
sns.distplot(diamonds.price, kde=False, color="b") #Bins based on scale so different shape if subset
plt.subplot(3,2,2)
sns.distplot(np.log(diamonds.price), kde=False, color="b",axlabel='Natural Log of Price')

plt.show()
# Release memory.
plt.clf()
plt.close()


# ### Add lprice column - log of price

# In[9]:

diamonds['lprice']=np.log(diamonds.price)


# In[85]:

#diamonds['log_carat']=np.log(diamonds.carat)
#diamonds.drop(diamonds.columns[[8]], axis=1,inplace=True)  # diamonds.columns is zero-based pd.Index 


# In[42]:

diamonds.describe()


# In[43]:

diamonds.head()


# #### Replot of Dataset

# In[ ]:

#After dropping out zero values we replot our data


# ### Price versus cut, color, clarity

# In[12]:

plt.figure(figsize=(12,10))
plt.subplot(3,3,1)
sns.violinplot(x="cut", y="price",data=diamonds,order=['Fair','Good','Very Good','Premium','Ideal'])
plt.subplot(3,3,2)
sns.violinplot(x='color',y='price',data=diamonds,order=['J','I','H','G','F','E', 'D'])
plt.subplot(3,3,3)
sns.violinplot(x='clarity',y='price',data=diamonds,order=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'])
plt.show()
# Release memory.
plt.clf()
plt.close()


# In[14]:

sns.pairplot(diamonds,x_vars=['carat','depth','table','x','y','z'],y_vars='price')

plt.show()
# Release memory.
plt.clf()
plt.close()


# In[30]:

sns.pairplot(diamonds,x_vars=['carat','depth','table'],y_vars='price',size = 7, aspect = 0.7,kind='reg') 
#line of best fit and 955 confidence
plt.show()
# Release memory.
plt.clf()
plt.close()


# In[23]:

sns.pairplot(diamonds,x_vars=['x','y','z'],y_vars='price',size = 7, aspect = 0.7,kind='reg') 
#line of best fit and 955 confidence

plt.show()
# Release memory.
plt.clf()
plt.close()


# In[24]:

sns.pairplot(diamonds,x_vars=['carat','depth','table'],y_vars='lprice',size = 7, aspect = 0.7,kind='reg') 
#line of best fit and 955 confidence
sns.pairplot(diamonds,x_vars=['x','y','z'],y_vars='lprice',size = 7, aspect = 0.7,kind='reg') 
#line of best fit and 955 confidence
plt.show()
# Release memory.
plt.clf()
plt.close()


# In[22]:

plt.figure(figsize=(12,10))
plt.subplot(3,3,1)
sns.distplot(diamonds.price, kde=False, color="b")
plt.subplot(3,3,2)
sns.distplot(diamonds.table, kde=False, color="b")
plt.subplot(3,3,3)
sns.distplot(diamonds.carat, kde=False, color="b")
plt.show()
# Release memory.
plt.clf()
plt.close()


# In[47]:

#plt.figure(figsize=(12,10))
#plt.subplot(2,2,1)
#sns.violinplot(x='price',y='x',data=diamonds)
#plt.subplot(2,2,2)
#sns.violinplot(x='price',y='y',data=diamonds)
#plt.subplot(2,2,3)
#sns.violinplot(x='price',y='z',data=diamonds)
#plt.subplot(2,2,4)
#sns.violinplot(x='price',y='z',data=diamonds)
#plt.show()
# Release memory.
#plt.clf()
#plt.close()


# In[44]:

plt.figure(figsize=(12,10))
plt.subplot(3,3,1)
sns.distplot(diamonds.price, kde=False, color="b")
plt.subplot(3,3,2)
sns.distplot(diamonds.lprice, kde=False, color="b")
plt.subplot(3,3,3)
sns.distplot(diamonds.carat, kde=False, color="b")
plt.show()
# Release memory.
plt.clf()
plt.close()


# In[73]:


#Take a look at carat and see if worth using log
#diamonds['lcarat']=np.log10(diamonds.carat)
#diamonds.columns[11]
#diamonds.drop(diamonds.columns[[11]], axis=1,inplace=True)  # diamonds.columns is zero-based pd.Index 
#drop it


# In[45]:

diamonds.describe()


# In[47]:

#https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python


plt.figure(figsize=(12,10))
plt.subplot(3,3,1)
sns.distplot(diamonds.price)
plt.subplot(3,3,2)
sns.distplot(diamonds.lprice)
plt.subplot(3,3,3)
sns.distplot(diamonds.carat)

plt.show()
# Release memory.
plt.clf()
plt.close()


# #### Take a closer look at the Distribution of lprice

# In[46]:

import scipy.stats as stats

plt.figure(figsize=(12,10))
plt.subplot(3,3,1)
#histogram and normal probability plot
sns.distplot(diamonds.lprice, fit=stats.norm)
#fig = plt.figure()
plt.subplot(3,3,2)
res = stats.probplot(diamonds.lprice, plot=plt)

plt.show()
# Release memory.
plt.clf()
plt.close()


# In[48]:

import scipy.stats as stats

fair_cut=diamonds[diamonds.cut=='Fair']
good_cut=diamonds[diamonds.cut=='Good']
vgood_cut=diamonds[diamonds.cut=='Very Good']
prem_cut=diamonds[diamonds.cut=='Premium']
ideal_cut=diamonds[diamonds.cut=='Ideal']   


plt.figure(figsize=(12,10))

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.9)

plt.subplot(5,2,1)
#histogram and normal probability plot
sns.axes_style({'legend.frameon': True})
#sns.set_style({'legend.frameon': True})
sns.distplot(fair_cut.lprice, fit=stats.norm,label='Fair',axlabel='lprice (Fair)')
plt.subplot(5,2,2)
res = stats.probplot(fair_cut.lprice, plot=plt)
#fig = plt.figure()
plt.subplot(5,2,3)
sns.distplot(good_cut.lprice, fit=stats.norm, label= 'Good',axlabel='lprice (Good)')
plt.subplot(5,2,4)
res = stats.probplot(good_cut.lprice, plot=plt)

plt.subplot(5,2,5)
sns.distplot(vgood_cut.lprice, fit=stats.norm, label= 'Very Good',axlabel='lprice (Very Good)')
plt.subplot(5,2,6)
res = stats.probplot(vgood_cut.lprice, plot=plt)

plt.subplot(5,2,7)
sns.distplot(prem_cut.lprice, fit=stats.norm, label= 'Premium',axlabel='lprice (Premium)')

plt.subplot(5,2,8)
res = stats.probplot(prem_cut.lprice, plot=plt)


plt.subplot(5,2,9)
sns.distplot(ideal_cut.lprice, fit=stats.norm, label= 'Ideal',axlabel='lprice (Ideal)') 

plt.subplot(5,2,10)
res = stats.probplot(ideal_cut.lprice, plot=plt)


plt.show()
# Release memory.
plt.clf()
plt.close()


# In[ ]:




# In[83]:

import scipy.stats as stats

fair_cut=diamonds[diamonds.cut=='Fair']
good_cut=diamonds[diamonds.cut=='Good']
vgood_cut=diamonds[diamonds.cut=='Very Good']
prem_cut=diamonds[diamonds.cut=='Premium']
ideal_cut=diamonds[diamonds.cut=='Ideal']   


plt.figure(figsize=(12,10))

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.9)

plt.subplot(5,2,1)
#histogram and normal probability plot
sns.axes_style({'legend.frameon': True})
#sns.set_style({'legend.frameon': True})
sns.distplot(fair_cut.carat, fit=stats.norm,label='Fair',axlabel='Carat (Fair)')
plt.subplot(5,2,2)
res = stats.probplot(fair_cut.carat, plot=plt)
#fig = plt.figure()
plt.subplot(5,2,3)
sns.distplot(good_cut.carat, fit=stats.norm, label= 'Good',axlabel='Carat (Good)')
plt.subplot(5,2,4)
res = stats.probplot(good_cut.carat, plot=plt)

plt.subplot(5,2,5)
sns.distplot(vgood_cut.carat, fit=stats.norm, label= 'Very Good',axlabel='Carat (Very Good)')
plt.subplot(5,2,6)
res = stats.probplot(vgood_cut.carat, plot=plt)

plt.subplot(5,2,7)
sns.distplot(prem_cut.carat, fit=stats.norm, label= 'Premium',axlabel='Carat (Premium)')

plt.subplot(5,2,8)
res = stats.probplot(prem_cut.carat, plot=plt)


plt.subplot(5,2,9)
sns.distplot(ideal_cut.carat, fit=stats.norm, label= 'Ideal',axlabel='Carat (Ideal)') 

plt.subplot(5,2,10)
res = stats.probplot(ideal_cut.carat, plot=plt)


plt.show()
# Release memory.
plt.clf()
plt.close()


# In[49]:

import scipy.stats as stats


I1_clarity=diamonds[diamonds.clarity=='I1']
SI2_clarity=diamonds[diamonds.clarity=='SI2']
SI1_clarity=diamonds[diamonds.clarity=='SI1']
VS2_clarity=diamonds[diamonds.clarity=='VS2']
VS1_clarity=diamonds[diamonds.clarity=='VS1']
VVS2_clarity=diamonds[diamonds.clarity=='VVS2']
VVS1_clarity=diamonds[diamonds.clarity=='VVS1']
IF_clarity=diamonds[diamonds.clarity=='IF']




plt.figure(figsize=(12,10))

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1.5)

plt.subplot(8,2,1)
#histogram and normal probability plot
sns.axes_style({'legend.frameon': True})
#sns.set_style({'legend.frameon': True})
sns.distplot(I1_clarity.lprice, fit=stats.norm,label='I1',axlabel='lprice (I1)')
plt.subplot(8,2,2)
res = stats.probplot(I1_clarity.lprice, plot=plt)
#fig = plt.figure()
plt.subplot(8,2,3)
sns.distplot(SI2_clarity.lprice, fit=stats.norm, label= 'SI2',axlabel='lprice (SI2)')
plt.subplot(8,2,4)
res = stats.probplot(SI2_clarity.lprice, plot=plt)

plt.subplot(8,2,5)
sns.distplot(SI1_clarity.lprice, fit=stats.norm, label= 'SI1',axlabel='lprice (SI1)')
plt.subplot(8,2,6)
res = stats.probplot(SI1_clarity.lprice, plot=plt)

plt.subplot(8,2,7)
sns.distplot(VS2_clarity.lprice, fit=stats.norm, label= 'VS2',axlabel='lprice (VS2)')

plt.subplot(8,2,8)
res = stats.probplot(VS2_clarity.lprice, plot=plt)


plt.subplot(8,2,9)
sns.distplot(VS1_clarity.lprice, fit=stats.norm, label= 'VS1',axlabel='lprice (VS1)') 

plt.subplot(8,2,10)
res = stats.probplot(VS1_clarity.lprice, plot=plt)


plt.subplot(8,2,11)
sns.distplot(VVS2_clarity.lprice, fit=stats.norm, label= 'VVS2',axlabel='lprice (VVS2)')

plt.subplot(8,2,12)
res = stats.probplot(VVS2_clarity.lprice, plot=plt)


plt.subplot(8,2,13)
sns.distplot(VVS1_clarity.lprice, fit=stats.norm, label= 'VVS1',axlabel='lprice (VVS1)') 

plt.subplot(8,2,14)
res = stats.probplot(VVS1_clarity.lprice, plot=plt)

plt.subplot(8,2,15)
sns.distplot(IF_clarity.lprice, fit=stats.norm, label= 'IF',axlabel='lprice (IF)') 

plt.subplot(8,2,16)
res = stats.probplot(IF_clarity.lprice, plot=plt)




plt.show()
# Release memory.
plt.clf()
plt.close()


# In[93]:

import scipy.stats as stats


I1_clarity=diamonds[diamonds.clarity=='I1']
SI2_clarity=diamonds[diamonds.clarity=='SI2']
SI1_clarity=diamonds[diamonds.clarity=='SI1']
VS2_clarity=diamonds[diamonds.clarity=='VS2']
VS1_clarity=diamonds[diamonds.clarity=='VS1']
VVS2_clarity=diamonds[diamonds.clarity=='VVS2']
VVS1_clarity=diamonds[diamonds.clarity=='VVS1']
IF_clarity=diamonds[diamonds.clarity=='IF']




plt.figure(figsize=(12,10))

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1.5)

plt.subplot(8,2,1)
#histogram and normal probability plot
sns.axes_style({'legend.frameon': True})
#sns.set_style({'legend.frameon': True})
sns.distplot(I1_clarity.carat, fit=stats.norm,label='I1',axlabel='carat (I1)')
plt.subplot(8,2,2)
res = stats.probplot(I1_clarity.carat, plot=plt)
#fig = plt.figure()
plt.subplot(8,2,3)
sns.distplot(SI2_clarity.carat, fit=stats.norm, label= 'SI2',axlabel='carat (SI2)')
plt.subplot(8,2,4)
res = stats.probplot(SI2_clarity.carat, plot=plt)

plt.subplot(8,2,5)
sns.distplot(SI1_clarity.carat, fit=stats.norm, label= 'SI1',axlabel='carate (SI1)')
plt.subplot(8,2,6)
res = stats.probplot(SI1_clarity.carat, plot=plt)

plt.subplot(8,2,7)
sns.distplot(VS2_clarity.carat, fit=stats.norm, label= 'VS2',axlabel='carat (VS2)')

plt.subplot(8,2,8)
res = stats.probplot(VS2_clarity.carat, plot=plt)


plt.subplot(8,2,9)
sns.distplot(VS1_clarity.carat, fit=stats.norm, label= 'VS1',axlabel='carat (VS1)') 

plt.subplot(8,2,10)
res = stats.probplot(VS1_clarity.carat, plot=plt)


plt.subplot(8,2,11)
sns.distplot(VVS2_clarity.carat, fit=stats.norm, label= 'VVS2',axlabel='lprice (VVS2)')

plt.subplot(8,2,12)
res = stats.probplot(VVS2_clarity.carat, plot=plt)


plt.subplot(8,2,13)
sns.distplot(VVS1_clarity.carat, fit=stats.norm, label= 'VVS1',axlabel='carat (VVS1)') 

plt.subplot(8,2,14)
res = stats.probplot(VVS1_clarity.carat, plot=plt)

plt.subplot(8,2,15)
sns.distplot(IF_clarity.carat, fit=stats.norm, label= 'IF',axlabel='carat (IF)') 

plt.subplot(8,2,16)
res = stats.probplot(IF_clarity.carat, plot=plt)




plt.show()
# Release memory.
plt.clf()
plt.close()


# In[24]:

plt.figure(figsize=(12,10))
sns.pairplot(diamonds)
plt.show()
# Release memory.
plt.clf()
plt.close()


# #### What's the Distribution in terms of the cut

# In[54]:

sns.countplot(x="cut", data=diamonds,order=['Fair','Good','Very Good','Premium','Ideal'])
plt.show()
# Release memory.
plt.clf()
plt.close()


# #### What about clarity?

# In[55]:

sns.countplot(x="clarity", data=diamonds,order=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'])
plt.show()
# Release memory.
plt.clf()
plt.close()


# #### Highlight the difference due to cut

# In[95]:


plt.figure(figsize=(1,1))

fig=diamonds[diamonds.cut=='Fair'].plot(kind='scatter',x='price',y='carat',color='orange', label='Fair')
diamonds[diamonds.cut=='Good'].plot(kind='scatter',x='price',y='carat',color='red', label='Good',ax=fig)
diamonds[diamonds.cut=='Very Good'].plot(kind='scatter',x='price',y='carat',color='magenta', label='Very Good', ax=fig)
diamonds[diamonds.cut=='Premium'].plot(kind='scatter',x='price',y='carat',color='green', label='Premium', ax=fig)
diamonds[diamonds.cut=='Ideal'].plot(kind='scatter',x='price',y='carat',color='blue', label='Ideal', ax=fig)
fig.set_xlabel("price")
fig.set_ylabel("carat")
fig.set_title("Price VS Carat Weight")

plt.legend(loc = 2) #Don't use loc = 0  as this positions legend  top right and causes confusion with data - move legend to top left



fig=diamonds[diamonds.cut=='Fair'].plot(kind='scatter',x='lprice',y='carat',color='orange', label='Fair')
diamonds[diamonds.cut=='Good'].plot(kind='scatter',x='lprice',y='carat',color='red', label='Good',ax=fig)
diamonds[diamonds.cut=='Very Good'].plot(kind='scatter',x='lprice',y='carat',color='magenta', label='Very Good', ax=fig)
diamonds[diamonds.cut=='Premium'].plot(kind='scatter',x='lprice',y='carat',color='green', label='Premium', ax=fig)
diamonds[diamonds.cut=='Ideal'].plot(kind='scatter',x='lprice',y='carat',color='blue', label='Ideal', ax=fig)
fig.set_xlabel("log price")
fig.set_ylabel("carat")
fig.set_title("Log Price VS Carat Weight")

plt.legend(loc = 2) #Don't use loc = 0  as this positions legend  top right and causes confusion with data - move legend to top left



plt.show()
# Release memory.
plt.clf()
plt.close()


# #### How does cut affect price  and lprice?

# In[110]:




#fig, ax = plt.subplots(figsize=(1, 1))

fig = diamonds[diamonds.cut=='Fair'].plot(kind='scatter',x='price',y='x',color='orange', label='Fair')
diamonds[diamonds.cut=='Good'].plot(kind='scatter',x='price',y='x',color='red', label='Good',ax=fig)
diamonds[diamonds.cut=='Very Good'].plot(kind='scatter',x='price',y='x',color='magenta', label='Very Good', ax=fig)
diamonds[diamonds.cut=='Premium'].plot(kind='scatter',x='price',y='x',color='green', label='Premium', ax=fig)
diamonds[diamonds.cut=='Ideal'].plot(kind='scatter',x='price',y='x',color='blue', label='Ideal', ax=fig)
fig.set_xlabel("price")
fig.set_ylabel("x")
fig.set_title("Price VS x")
plt.legend(loc = 2) #Don't use loc = 0  as this positions legend  top right and causes confusion with data - move legend to top left




fig = diamonds[diamonds.cut=='Fair'].plot(kind='scatter',x='lprice',y='x',color='orange', label='Fair')
diamonds[diamonds.cut=='Good'].plot(kind='scatter',x='lprice',y='x',color='red', label='Good',ax=fig)
diamonds[diamonds.cut=='Very Good'].plot(kind='scatter',x='lprice',y='x',color='magenta', label='Very Good', ax=fig)
diamonds[diamonds.cut=='Premium'].plot(kind='scatter',x='lprice',y='x',color='green', label='Premium', ax=fig)
diamonds[diamonds.cut=='Ideal'].plot(kind='scatter',x='lprice',y='x',color='blue', label='Ideal', ax=fig)
fig.set_xlabel("log price")
fig.set_ylabel("x")
fig.set_title("Log Price VS x")
plt.legend(loc = 2) #Don't use loc = 0  as this positions legend  top right and causes confusion with data - move legend to top left


fig = diamonds[diamonds.cut=='Fair'].plot(kind='scatter',x='price',y='y',color='orange', label='Fair')
diamonds[diamonds.cut=='Good'].plot(kind='scatter',x='price',y='y',color='red', label='Good',ax=fig)
diamonds[diamonds.cut=='Very Good'].plot(kind='scatter',x='price',y='y',color='magenta', label='Very Good', ax=fig)
diamonds[diamonds.cut=='Premium'].plot(kind='scatter',x='price',y='y',color='green', label='Premium', ax=fig)
diamonds[diamonds.cut=='Ideal'].plot(kind='scatter',x='price',y='y',color='blue', label='Ideal', ax=fig)
fig.set_xlabel("price")
fig.set_ylabel("y")
fig.set_title("Price VS y")
plt.legend(loc = 2) #Don't use loc = 0  as this positions legend  top right and causes confusion with data - move legend to top left


fig = diamonds[diamonds.cut=='Fair'].plot(kind='scatter',x='lprice',y='y',color='orange', label='Fair')
diamonds[diamonds.cut=='Good'].plot(kind='scatter',x='lprice',y='y',color='red', label='Good',ax=fig)
diamonds[diamonds.cut=='Very Good'].plot(kind='scatter',x='lprice',y='y',color='magenta', label='Very Good', ax=fig)
diamonds[diamonds.cut=='Premium'].plot(kind='scatter',x='lprice',y='y',color='green', label='Premium', ax=fig)
diamonds[diamonds.cut=='Ideal'].plot(kind='scatter',x='lprice',y='y',color='blue', label='Ideal', ax=fig)
fig.set_xlabel("log price")
fig.set_ylabel("y")
fig.set_title("Log Price VS y")
plt.legend(loc = 2) #Don't use loc = 0  as this positions legend  top right and causes confusion with data - move legend to top left


fig = diamonds[diamonds.cut=='Fair'].plot(kind='scatter',x='price',y='z',color='orange', label='Fair')
diamonds[diamonds.cut=='Good'].plot(kind='scatter',x='price',y='z',color='red', label='Good',ax=fig)
diamonds[diamonds.cut=='Very Good'].plot(kind='scatter',x='price',y='z',color='magenta', label='Very Good', ax=fig)
diamonds[diamonds.cut=='Premium'].plot(kind='scatter',x='price',y='z',color='green', label='Premium', ax=fig)
diamonds[diamonds.cut=='Ideal'].plot(kind='scatter',x='price',y='z',color='blue', label='Ideal', ax=fig)
fig.set_xlabel("price")
fig.set_ylabel("z")
fig.set_title("Price VS z")
plt.legend(loc = 2) #Don't use loc = 0  as this positions legend  top right and causes confusion with data - move legend to top left


fig = diamonds[diamonds.cut=='Fair'].plot(kind='scatter',x='lprice',y='z',color='orange', label='Fair')
diamonds[diamonds.cut=='Good'].plot(kind='scatter',x='lprice',y='z',color='red', label='Good',ax=fig)
diamonds[diamonds.cut=='Very Good'].plot(kind='scatter',x='lprice',y='z',color='magenta', label='Very Good', ax=fig)
diamonds[diamonds.cut=='Premium'].plot(kind='scatter',x='lprice',y='z',color='green', label='Premium', ax=fig)
diamonds[diamonds.cut=='Ideal'].plot(kind='scatter',x='lprice',y='z',color='blue', label='Ideal', ax=fig)
fig.set_xlabel("log price")
fig.set_ylabel("z")
fig.set_title("Log Price VS z")
plt.legend(loc = 2) #Don't use loc = 0  as this positions legend  top right and causes confusion with data - move legend to top left





plt.show()
# Release memory.
plt.clf()
plt.close()


# #### Breakout Premium and Ideal - does this make things any clearer?

# In[111]:


fig = diamonds[diamonds.cut=='Premium'].plot(kind='scatter',x='price',y='carat',color='green', label='Premium')
diamonds[diamonds.cut=='Ideal'].plot(kind='scatter',x='price',y='carat',color='blue', label='Ideal', ax=fig)
fig.set_xlabel("price")
fig.set_ylabel("carat")
fig.set_title("Price VS Carat Weight")
plt.legend(loc = 2) #Don't use loc = 0  as this positions legend  top right and causes confusion with data - move legend to top left


fig = diamonds[diamonds.cut=='Premium'].plot(kind='scatter',x='lprice',y='carat',color='green', label='Premium')
diamonds[diamonds.cut=='Ideal'].plot(kind='scatter',x='lprice',y='carat',color='blue', label='Ideal', ax=fig)
fig.set_xlabel("lprice")
fig.set_ylabel("carat")
fig.set_title("Log Price VS Carat Weight")
plt.legend(loc = 2) #Don't use loc = 0  as this positions legend  top right and causes confusion with data - move legend to top left

plt.show()
# Release memory.
plt.clf()
plt.close()


# #### Breakout the other cut types - less desirable?

# In[112]:

fig = diamonds[diamonds.cut=='Fair'].plot(kind='scatter',x='price',y='carat',color='orange', label='Fair')
diamonds[diamonds.cut=='Good'].plot(kind='scatter',x='price',y='carat',color='red', label='Good',ax=fig)
diamonds[diamonds.cut=='Very Good'].plot(kind='scatter',x='price',y='carat',color='magenta', label='Very Good', ax=fig)
fig.set_xlabel("price")
fig.set_ylabel("carat")
fig.set_title("Price VS Carat Weight")
plt.legend(loc = 2) #Don't use loc = 0  as this positions legend  top right and causes confusion with data - move legend to top left

fig = diamonds[diamonds.cut=='Fair'].plot(kind='scatter',x='lprice',y='carat',color='orange', label='Fair')
diamonds[diamonds.cut=='Good'].plot(kind='scatter',x='lprice',y='carat',color='red', label='Good',ax=fig)
diamonds[diamonds.cut=='Very Good'].plot(kind='scatter',x='lprice',y='carat',color='magenta', label='Very Good', ax=fig)
fig.set_xlabel("lprice")
fig.set_ylabel("carat")
fig.set_title("Log Price VS Carat Weight")
plt.legend(loc = 2) #Don't use loc = 0  as this positions legend  top right and causes confusion with data - move legend to top left


plt.show()
# Release memory.
plt.clf()
plt.close()


# #### Diamond depth and table against pricing for all cuts

# In[40]:

fig = diamonds[diamonds.cut=='Fair'].plot(kind='scatter',x='price',y='depth',color='orange', label='Fair')
diamonds[diamonds.cut=='Good'].plot(kind='scatter',x='price',y='depth',color='red', label='Good',ax=fig)
diamonds[diamonds.cut=='Very Good'].plot(kind='scatter',x='price',y='depth',color='magenta', label='Very Good', ax=fig)
diamonds[diamonds.cut=='Premium'].plot(kind='scatter',x='price',y='depth',color='green', label='Premium', ax=fig)
diamonds[diamonds.cut=='Ideal'].plot(kind='scatter',x='price',y='depth',color='blue', label='Ideal', ax=fig)
fig.set_xlabel("price")
fig.set_ylabel("depth")
fig.set_title("Price VS Depth")
plt.legend(loc = 2) #Don't use loc = 0  as this positions legend  top right and causes confusion with data - move legend to top left

fig = diamonds[diamonds.cut=='Fair'].plot(kind='scatter',x='lprice',y='depth',color='orange', label='Fair')
diamonds[diamonds.cut=='Good'].plot(kind='scatter',x='lprice',y='depth',color='red', label='Good',ax=fig)
diamonds[diamonds.cut=='Very Good'].plot(kind='scatter',x='lprice',y='depth',color='magenta', label='Very Good', ax=fig)
diamonds[diamonds.cut=='Premium'].plot(kind='scatter',x='lprice',y='depth',color='green', label='Premium', ax=fig)
diamonds[diamonds.cut=='Ideal'].plot(kind='scatter',x='lprice',y='depth',color='blue', label='Ideal', ax=fig)
fig.set_xlabel("lprice")
fig.set_ylabel("depth")
fig.set_title("Log Price VS Depth")
plt.legend(loc = 2) #Don't use loc = 0  as this positions legend  top right and causes confusion with data - move legend to top left



plt.show()
# Release memory.
plt.clf()
plt.close()


# In[72]:

fig = diamonds[diamonds.cut=='Fair'].plot(kind='scatter',x='price',y='table',color='orange', label='Fair')
diamonds[diamonds.cut=='Good'].plot(kind='scatter',x='price',y='table',color='red', label='Good',ax=fig)
diamonds[diamonds.cut=='Very Good'].plot(kind='scatter',x='price',y='table',color='magenta', label='Very Good', ax=fig)
diamonds[diamonds.cut=='Premium'].plot(kind='scatter',x='price',y='table',color='green', label='Premium', ax=fig)
diamonds[diamonds.cut=='Ideal'].plot(kind='scatter',x='price',y='table',color='blue', label='Ideal', ax=fig)
fig.set_xlabel("price")
fig.set_ylabel("table")
fig.set_title("Price VS Table")
plt.legend(loc = 2) #Don't use loc = 0  as this positions legend  top right and causes confusion with data - move legend to top left

fig = diamonds[diamonds.cut=='Fair'].plot(kind='scatter',x='lprice',y='table',color='orange', label='Fair')
diamonds[diamonds.cut=='Good'].plot(kind='scatter',x='lprice',y='table',color='red', label='Good',ax=fig)
diamonds[diamonds.cut=='Very Good'].plot(kind='scatter',x='lprice',y='table',color='magenta', label='Very Good', ax=fig)
diamonds[diamonds.cut=='Premium'].plot(kind='scatter',x='lprice',y='table',color='green', label='Premium', ax=fig)
diamonds[diamonds.cut=='Ideal'].plot(kind='scatter',x='lprice',y='table',color='blue', label='Ideal', ax=fig)
fig.set_xlabel("lprice")
fig.set_ylabel("table")
fig.set_title("Log Price VS Table")
plt.legend(loc = 2) #Don't use loc = 0  as this positions legend  top right and causes confusion with data - move legend to top left


plt.show()

# Release memory.
plt.clf()
plt.close()


# #### Can we see anything with the most desirable cut - Ideal

# In[73]:

fig = diamonds[diamonds.cut=='Ideal'].plot(kind='scatter',x='price',y='carat',color='blue', label='Ideal')
fig.set_xlabel("price")
fig.set_ylabel("carat")
fig.set_title("Price VS Carat")
plt.legend(loc = 2) #Don't use loc = 0  as this positions legend  top right and causes confusion with data - move legend to top left
plt.show()

fig = diamonds[diamonds.cut=='Ideal'].plot(kind='scatter',x='lprice',y='carat',color='blue', label='Ideal')
fig.set_xlabel("lprice")
fig.set_ylabel("carat")
fig.set_title("Log Price VS Carat")
plt.legend(loc = 2) #Don't use loc = 0  as this positions legend  top right and causes confusion with data - move legend to top left
plt.show()

# Release memory.
plt.clf()
plt.close()


# #### Can we see any difference when we use Clarity 

# In[74]:

fig = diamonds[diamonds.clarity=='I1'].plot(kind='scatter',x='price',y='carat',color='orange', label='I1')
diamonds[diamonds.clarity=='SI2'].plot(kind='scatter',x='price',y='carat',color='red', label='SI2',ax=fig)
diamonds[diamonds.clarity=='SI1'].plot(kind='scatter',x='price',y='carat',color='magenta', label='SI1', ax=fig)
diamonds[diamonds.clarity=='VS2'].plot(kind='scatter',x='price',y='carat',color='green', label='VS2', ax=fig)
diamonds[diamonds.clarity=='VS1'].plot(kind='scatter',x='price',y='carat',color='blue', label='VS1', ax=fig)
diamonds[diamonds.clarity=='VVS2'].plot(kind='scatter',x='price',y='carat',color='red', label='VVS2',ax=fig)
diamonds[diamonds.clarity=='VVS1'].plot(kind='scatter',x='price',y='carat',color='magenta', label='VVS1', ax=fig)
diamonds[diamonds.clarity=='IF'].plot(kind='scatter',x='price',y='carat',color='green', label='IF', ax=fig)

fig.set_xlabel("price")
fig.set_ylabel("carat")
fig.set_title("Price VS Carat Weight")
plt.legend(loc = 2)


fig = diamonds[diamonds.clarity=='I1'].plot(kind='scatter',x='lprice',y='carat',color='orange', label='I1')
diamonds[diamonds.clarity=='SI2'].plot(kind='scatter',x='lprice',y='carat',color='red', label='SI2',ax=fig)
diamonds[diamonds.clarity=='SI1'].plot(kind='scatter',x='lprice',y='carat',color='magenta', label='SI1', ax=fig)
diamonds[diamonds.clarity=='VS2'].plot(kind='scatter',x='lprice',y='carat',color='green', label='VS2', ax=fig)
diamonds[diamonds.clarity=='VS1'].plot(kind='scatter',x='lprice',y='carat',color='blue', label='VS1', ax=fig)
diamonds[diamonds.clarity=='VVS2'].plot(kind='scatter',x='lprice',y='carat',color='red', label='VVS2',ax=fig)
diamonds[diamonds.clarity=='VVS1'].plot(kind='scatter',x='lprice',y='carat',color='magenta', label='VVS1', ax=fig)
diamonds[diamonds.clarity=='IF'].plot(kind='scatter',x='lprice',y='carat',color='green', label='IF', ax=fig)

fig.set_xlabel("lprice")
fig.set_ylabel("carat")
fig.set_title("Log Price VS Carat Weight")
plt.legend(loc = 2)


plt.show()

# Release memory.
plt.clf()
plt.close()


# In[113]:

fig = diamonds[diamonds.clarity=='I1'].plot(kind='scatter',x='price',y='carat',color='orange', label='I1')
diamonds[diamonds.clarity=='SI2'].plot(kind='scatter',x='price',y='carat',color='red', label='SI2',ax=fig)
diamonds[diamonds.clarity=='SI1'].plot(kind='scatter',x='price',y='carat',color='magenta', label='SI1', ax=fig)
diamonds[diamonds.clarity=='VS2'].plot(kind='scatter',x='price',y='carat',color='green', label='VS2', ax=fig)
diamonds[diamonds.clarity=='VS1'].plot(kind='scatter',x='price',y='carat',color='blue', label='VS1', ax=fig)


fig.set_xlabel("price")
fig.set_ylabel("carat")
fig.set_title("Price VS Carat Weight")
plt.legend(loc = 2)


fig = diamonds[diamonds.clarity=='I1'].plot(kind='scatter',x='lprice',y='carat',color='orange', label='I1')
diamonds[diamonds.clarity=='SI2'].plot(kind='scatter',x='lprice',y='carat',color='red', label='SI2',ax=fig)
diamonds[diamonds.clarity=='SI1'].plot(kind='scatter',x='lprice',y='carat',color='magenta', label='SI1', ax=fig)
diamonds[diamonds.clarity=='VS2'].plot(kind='scatter',x='lprice',y='carat',color='green', label='VS2', ax=fig)
diamonds[diamonds.clarity=='VS1'].plot(kind='scatter',x='lprice',y='carat',color='blue', label='VS1', ax=fig)


fig.set_xlabel("lprice")
fig.set_ylabel("carat")
fig.set_title("Log Price VS Carat Weight")
plt.legend(loc = 2)


plt.show()

# Release memory.
plt.clf()
plt.close()


# #### Break out the highest grades for clarity

# In[16]:

fig = diamonds[diamonds.clarity=='VVS2'].plot(kind='scatter',x='price',y='carat',color='red', label='VVS2')
diamonds[diamonds.clarity=='VVS1'].plot(kind='scatter',x='price',y='carat',color='magenta', label='VVS1', ax=fig)
diamonds[diamonds.clarity=='IF'].plot(kind='scatter',x='price',y='carat',color='green', label='IF', ax=fig)

fig.set_xlabel("price")
fig.set_ylabel("carat")
fig.set_title("Price VS Carat Weight")
plt.legend(loc = 2)


fig = diamonds[diamonds.clarity=='VVS2'].plot(kind='scatter',x='lprice',y='carat',color='red', label='VVS2')
diamonds[diamonds.clarity=='VVS1'].plot(kind='scatter',x='lprice',y='carat',color='magenta', label='VVS1', ax=fig)
diamonds[diamonds.clarity=='IF'].plot(kind='scatter',x='lprice',y='carat',color='green', label='IF', ax=fig)

fig.set_xlabel("lprice")
fig.set_ylabel("carat")
fig.set_title("Log Price VS Carat Weight")
plt.legend(loc = 2)


plt.show()

# Release memory.
plt.clf()
plt.close()


# In[54]:



fig = diamonds[diamonds.cut=='Fair'].plot(kind='density',x='lprice',y='carat',color='orange', label='Fair')
diamonds[diamonds.cut=='Good'].plot(kind='density',x='lprice',y='carat',color='red', label='Good',ax=fig)
diamonds[diamonds.cut=='Very Good'].plot(kind='density',x='lprice',y='carat',color='magenta', label='Very Good', ax=fig)
diamonds[diamonds.cut=='Premium'].plot(kind='density',x='lprice',y='carat',color='green', label='Premium', ax=fig)
diamonds[diamonds.cut=='Ideal'].plot(kind='density',x='lprice',y='carat',color='blue', label='Ideal', ax=fig)
fig.set_xlabel("price")
fig.set_ylabel("carat")
fig.set_title("Density Plot Log Price VS Carat Weight")
plt.legend(loc = 2) #Don't use loc = 0  as this positions legend  top right and causes confusion with data - move legend to top left



fig = diamonds[diamonds.cut=='Fair'].plot(kind='density',x='lprice',y='depth',color='orange', label='Fair')
diamonds[diamonds.cut=='Good'].plot(kind='density',x='lprice',y='depth',color='red', label='Good',ax=fig)
diamonds[diamonds.cut=='Very Good'].plot(kind='density',x='lprice',y='depth',color='magenta', label='Very Good', ax=fig)
diamonds[diamonds.cut=='Premium'].plot(kind='density',x='lprice',y='depth',color='green', label='Premium', ax=fig)
diamonds[diamonds.cut=='Ideal'].plot(kind='density',x='lprice',y='depth',color='blue', label='Ideal', ax=fig)
fig.set_xlabel("lprice")
fig.set_ylabel("depth")
fig.set_title("Density Plot Log Price VS Depth")
plt.legend(loc = 2) #Don't use loc = 0  as this positions legend  top right and causes confusion with data - move legend to top left



fig = diamonds[diamonds.clarity=='I1'].plot(kind='density',x='lprice',y='carat',color='orange', label='I1')
diamonds[diamonds.clarity=='SI2'].plot(kind='density',x='lprice',y='carat',color='red', label='SI2',ax=fig)
diamonds[diamonds.clarity=='SI1'].plot(kind='density',x='lprice',y='carat',color='magenta', label='SI1', ax=fig)
diamonds[diamonds.clarity=='VS2'].plot(kind='density',x='lprice',y='carat',color='green', label='VS2', ax=fig)
diamonds[diamonds.clarity=='VS1'].plot(kind='density',x='lprice',y='carat',color='blue', label='VS1', ax=fig)
diamonds[diamonds.clarity=='VVS2'].plot(kind='density',x='lprice',y='carat',color='red', label='VVS2',ax=fig)
diamonds[diamonds.clarity=='VVS1'].plot(kind='density',x='lprice',y='carat',color='magenta', label='VVS1', ax=fig)
diamonds[diamonds.clarity=='IF'].plot(kind='density',x='lprice',y='carat',color='green', label='IF', ax=fig)

fig.set_xlabel("lprice")
fig.set_ylabel("carat")
fig.set_title("Density Plot Log Price VS Carat Weight")
plt.legend(loc = 2)



fig = diamonds[diamonds.clarity=='I1'].plot(kind='density',x='lprice',y='depth',color='orange', label='I1')
diamonds[diamonds.clarity=='SI2'].plot(kind='density',x='lprice',y='depth',color='red', label='SI2',ax=fig)
diamonds[diamonds.clarity=='SI1'].plot(kind='density',x='lprice',y='depth',color='magenta', label='SI1', ax=fig)
diamonds[diamonds.clarity=='VS2'].plot(kind='density',x='lprice',y='depth',color='green', label='VS2', ax=fig)
diamonds[diamonds.clarity=='VS1'].plot(kind='density',x='lprice',y='depth',color='blue', label='VS1', ax=fig)
diamonds[diamonds.clarity=='VVS2'].plot(kind='density',x='lprice',y='depth',color='red', label='VVS2',ax=fig)
diamonds[diamonds.clarity=='VVS1'].plot(kind='density',x='lprice',y='depth',color='magenta', label='VVS1', ax=fig)
diamonds[diamonds.clarity=='IF'].plot(kind='density',x='lprice',y='depth',color='green', label='IF', ax=fig)

fig.set_xlabel("lprice")
fig.set_ylabel("carat")
fig.set_title("Density Plot Log Price VS Carat Weight")
plt.legend(loc = 2)


plt.show()

# Release memory.
plt.clf()
plt.close()


# In[75]:

diamonds.color.unique()


# In[76]:

#(I1 Included (worst), SI2 (Slightly Included grade 2), SI1, VS2 (very slightly included grade 2), VS1, 
#VVS2 (very very slightly included grade 2), VVS1, IF (Internally Flawless)(best))
diamonds.clarity.unique()


# ### Add a flag for Ideal cut only - Ideal= True, all others = False

# In[16]:

#df1['col3'] = df1['col2'].apply(lambda x: 1 if x > 0 else 0)
diamonds['ideal_flag'] = diamonds['cut'].apply(lambda x: 'True' if x == 'Ideal' else 'False')


# In[98]:

diamonds.columns[11] #Find ideal_flag - uppercase drop


# In[69]:

#use to drop dupicate flag column
#diamonds.drop(diamonds.columns[[11]], axis=1,inplace=True)  # diamonds.columns is zero-based pd.Index 
#drop it


# In[50]:

diamonds.describe()


# In[51]:

diamonds.head(5)


# In[71]:


fig = diamonds[diamonds.ideal_flag=='True'].plot(kind='scatter',x='price',y='carat',color='blue', label='True')
fig.set_xlabel("price")
fig.set_ylabel("carat")
fig.set_title("Price VS Carat")
plt.legend(loc = 2) #Don't use loc = 0  as this positions legend  top right and causes confusion with data - move legend to top left

plt.show()


fig = diamonds[diamonds.ideal_flag=='True'].plot(kind='scatter',x='lprice',y='carat',color='blue', label='True')
fig.set_xlabel("lprice")
fig.set_ylabel("carat")
fig.set_title("Log Price VS Carat")
plt.legend(loc = 2) #Don't use loc = 0  as this positions legend  top right and causes confusion with data - move legend to top left

plt.show()

# Release memory.
plt.clf()
plt.close()


# #### Use te Ideal flag to see if we can see any difference with Price per Carat

# In[54]:

fig = diamonds[diamonds.ideal_flag=='True'].plot(kind='scatter',x='price',y='carat',color='blue', label='True')
diamonds[diamonds.ideal_flag=='False'].plot(kind='scatter',x='price',y='carat',color='red', label='False',ax=fig)
fig.set_xlabel("price")
fig.set_ylabel("carat")
fig.set_title("Price VS Carat")
plt.legend(loc = 2) #Don't use 0 Best as causes confusion in top right move legend to top left

fig = diamonds[diamonds.ideal_flag=='True'].plot(kind='scatter',x='lprice',y='carat',color='blue', label='True')
diamonds[diamonds.ideal_flag=='False'].plot(kind='scatter',x='lprice',y='carat',color='red', label='False',ax=fig)
fig.set_xlabel("lprice")
fig.set_ylabel("carat")
fig.set_title("Log Price VS Carat")
plt.legend(loc = 2) #Don't use 0 Best as causes confusion in top right move legend to top left


plt.show()


# #### How does clarity and colour differ beween Ideal True and False

# In[115]:

plt.figure(figsize=(12,10))


plt.subplot(2,2,1)

plt.gca().set_ylim(0,9000)
sns.plt.title('Ideal Cut')

sns.countplot(x="clarity", data=diamonds[diamonds.ideal_flag=='True'],order=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'])


plt.subplot(2,2,2)
sns.plt.title('Not Ideal Cut')

sns.countplot(x="clarity", data=diamonds[diamonds.ideal_flag=='False'],order=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'])

plt.show()


plt.figure(figsize=(12,10))

plt.subplot(2,2,3)

plt.gca().set_ylim(0,9000)

sns.plt.title('Ideal Cut')

sns.countplot(x="color",data=diamonds[diamonds.ideal_flag=='True'],order=['D','E','F','G','H','I','J'])


plt.subplot(2,2,4)

plt.gca().set_ylim(0,9000)
sns.plt.title('Not Ideal Cut')

sns.countplot(x="color", data=diamonds[diamonds.ideal_flag=='False'],order=['D','E','F','G','H','I','J'])

plt.show()

# Release memory.
plt.clf()
plt.close()





# In[116]:

plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
plt.gca().set_ylim(0,7000)
sns.plt.title('Ideal Cut')

sns.countplot(x="color",data=diamonds[diamonds.ideal_flag=='True'],order=['D','E','F','G','H','I','J'])


plt.subplot(2,2,2)

sns.plt.title('Not Ideal Cut')

sns.countplot(x="color", data=diamonds[diamonds.ideal_flag=='False'],order=['D','E','F','G','H','I','J'])

plt.show()

# Release memory.
plt.clf()
plt.close()


# In[55]:

#Convert Pandas dataframe to numyyMatrix
numpyMatrix = diamonds.as_matrix()
numpyMatrix


# In[63]:

sns.jointplot(x="price", y="carat", data=diamonds, kind="kde");
plt.show()


# ### Correlations - Pearson

# In[52]:

diamonds.corr()


# In[53]:

plt.figure(figsize=(7,4)) 
sns.heatmap(diamonds.corr(),annot=True) #draws  heatmap with input as the correlation matrix calculted by(diamonds.corr())
plt.show()

# Release memory.
plt.clf()
plt.close()


# In[4]:

diamonds.corr()


# In[80]:

corr = diamonds.corr()
sns.pairplot(corr)
plt.show()


# In[ ]:

import seaborn as sns
sns.set()
sns.pairplot(diamonds, hue="lprice")


# In[65]:

#plt.figure(figsize=(12,10))
#plt.subplot(2,2,1)
#sns.distplot(diamonds.price)
#plt.subplot(2,2,2)
#sns.violinplot(x='price',y='y',data=diamonds)
#plt.subplot(2,2,3)
#sns.violinplot(x='price',y='z',data=diamonds)
#plt.subplot(2,2,4)
#sns.violinplot(x='price',y='z',data=diamonds)
#plt.show()


# ### Build some Models

# In[4]:

#based on https://www.analyticsvidhya.com/blog/2016/07/practical-guide-data-preprocessing-python-scikit-learn/

# importing alll the necessary packages to use the various classification algorithms
from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm
from sklearn.cross_validation import train_test_split #to split the dataset for training and testing
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm

seed=(1976)


# In[122]:

#Old train/test split - No longer used

#diamonds_train, diamonds_test = train_test_split(diamonds, test_size = 0.3)# in this our main data is split into train and test
# the attribute test_size=0.3 splits the data into 70% and 30% ratio. train=70% and test=30%
#print(diamonds_train.shape)
#print(diamonds_test.shape)


# In[4]:

#Fancy train/test split

from sklearn.cross_validation import train_test_split

#X_all = diamonds.drop(['ideal_flag','cut'], axis=1)
X_all = diamonds.drop(['ideal_flag'], axis=1)
Y_all = diamonds['ideal_flag']

num_test = 0.30
#train_X, test_X, train_y, test_y = train_test_split(X_all, Y_all, test_size=num_test, random_state=23)
diamonds_train, diamonds_test = train_test_split(diamonds,test_size=num_test,random_state=23)


# In[54]:

#Some Final Encoding based on #https://www.kaggle.com/enerrio/scikit-learn-ml-from-start-to-finish

#The last part of the preprocessing phase is to normalize labels. 
#The LabelEncoder in Scikit-learn will convert each unique string value into a number, 
#making out data more flexible for various algorithms.

#The result is a table of numbers that looks scary to humans, but beautiful to machines.

from sklearn import preprocessing
def encode_features(df_train, df_test):
    features = ['cut','clarity','color']
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test
    
diamonds_train, diamonds_test = encode_features(diamonds_train, diamonds_test)

diamonds_train.head()


# In[67]:

print(diamonds_train.shape)
print(diamonds_test.shape)

diamonds_train.describe()


# In[55]:

diamonds_test.describe()


# In[ ]:




# In[9]:

#Set up our training and test data
train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
train_y=diamonds_train[['lprice']] # output of our training data
test_X =diamonds_test[['carat','cut','clarity','color','x','y','z','depth','table']] # taking test data features
test_y=diamonds_test[['lprice']]  #output value of test data



# In[11]:

print(train_X.shape)
print(train_y.shape)

train_X.describe()

print(test_X.shape)
print(test_y.shape)

test_X.describe()


# In[56]:

test_y.describe()


# In[89]:

str(diamonds_norm_use)


# # GradientBoostRegressor - Variable importance

# In[57]:


#Based on following:
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD 3 clause


from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn import ensemble
from sklearn import datasets

#We'll need a normalised dataset later on so lets concatenate test and train  here
diamonds_norm = None
diamonds_norm = pd.concat([diamonds_train, diamonds_test], ignore_index=True)
diamonds_target=diamonds_norm['lprice'] # #Give diamonds.target the target value lprice

#Let indexes_to_drop be an array of positional indexes to drop ([0,1,2,3,4,5,6,7,8,9,11] in our case everything but lprice).
diamonds_target_use = None

#indexes_to_drop =[0,1,2,3,4,5,6,7,8,9,11]
#indexes_to_keep = set(range(diamonds_norm.shape[1])) - set(indexes_to_drop)
#diamonds_target_use = diamonds_norm.take(:list((indexes_to_keep)))

diamonds_target_use = diamonds_norm.iloc[:,[10]] #lprice

#We don't want to  use the idela_flag or the price so lets drop them
diamonds_norm_use = None

#Let indexes_to_drop be an array of positional indexes to drop ([6,11] in our case).
#indexes_to_drop =[11,6]
#indexes_to_keep = set(range(diamonds_norm.shape[1])) - set(indexes_to_drop)
#diamonds_norm_use = diamonds_norm.take(:list((indexes_to_keep)))

diamonds_norm_use = diamonds_norm.iloc[:,[0,1,2,3,4,5,7,8,9,10]] # All out other columns excluding price and ideal_flag


#diamonds_target_use = diamonds_norm.columns.take(list[0])
#diamonds_norm_use = diamonds_norm.drop(diamonds_norm.columns[[11]], axis=1,inplace=True) #Drop ideal_flag
#diamonds_norm_use = diamonds_norm.drop(diamonds_norm.columns[[6]], axis=1,inplace=True) #Drop price 

#We'll need a matrix later
#diamonds_norm_matrix = diamonds_norm_use.as_matrix()
#diamonds_target_matrix = pd.DataFrame(diamonds_target).as_matrix()

#pd.DataFrame(diamonds_norm.columns.take(list((indexes_to_keep))))

#numpyMatrix = df.as_matrix()

diamonds_norm_use = diamonds_norm.iloc[:,[0,1,2,3,4,5,7,8,9]] # All out other columns excluding lprice



X, y = shuffle(diamonds_norm_use, diamonds_target_use, random_state=23)
X = X.astype(np.float32)



offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

names = list(X_train)


X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)



#Reshape out Array from (37741L, 1L)) to 37741L,
y_train = np.reshape(y_train, -1)
y_test = np.reshape(y_test, -1)

feature_names = np.array(list(diamonds_norm_use))

#feature_names = np.array('carat', 'cut', 'clarity', 'color', 'x', 'y', 'z', 'depth', 'table')

#Fit regression model
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf.predict(X_test))
print("MSE: %.6f" % mse)


# Ok to here but Issue with clf.loss_


#Quick check to see we actually have something
#y_pred = clf.predict(X_test)

# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

#We have n_estimators': 500 so should have 500 scores
for i, y_pred in enumerate(clf.staged_predict(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)

#plt.figure(figsize=(40, 20))
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')


feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, feature_names[sorted_idx])
#plt.yticks(pos, boston.feature_names[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')


# # sort importances
# indices = np.argsort(clf.feature_importances_)
# # plot as bar chart
# plt.barh(np.arange(len(feature_names)), clf.feature_importances_[indices])
# plt.yticks(np.arange(len(feature_names)) + 0.25, np.array(feature_names)[indices])
# _ = plt.xlabel('Relative importance')
# #plt.savefig("Liberty_Feature_Importance.png")



plt.show()
# Release memory.
plt.clf()
plt.close()






# In[18]:

print(feature_importance[sorted_idx]),feature_names[sorted_idx][::-1]


# In[10]:

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn import datasets


diamonds_norm = None
diamonds_norm = pd.concat([diamonds_train, diamonds_test], ignore_index=True)
diamonds_target=diamonds_norm['lprice'] # #Give diamonds.target the target value lprice

#Let indexes_to_drop be an array of positional indexes to drop ([0,1,2,3,4,5,6,7,8,9,11] in our case everything but lprice).
diamonds_target_use = None
diamonds_target_use = diamonds_norm.iloc[:,[10]] #lprice

#We don't want to  use the idela_flag or the price so lets drop them
diamonds_norm_use = None

#Let indexes_to_drop be an array of positional indexes to drop ([6,11] in our case).


diamonds_norm_use = diamonds_norm.iloc[:,[0,1,2,3,4,5,7,8,9,10]] # All out other columns excluding price and ideal_flag


diamonds_norm_use = diamonds_norm.iloc[:,[0,1,2,3,4,5,7,8,9]] # All out other columns excluding lprice



X, y = shuffle(diamonds_norm_use, diamonds_target_use, random_state=23)
X = X.astype(np.float32)



offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

names = list(X_train)


X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)



#Reshape out Array from (37741L, 1L)) to 37741L,
y_train = np.reshape(y_train, -1)
#y_test = np.reshape(y_test, -1)

feature_names = np.array(list(diamonds_norm_use))


#Fit regression model
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

original_params = {'n_estimators': 1000, 'max_depth': None, 'random_state': 2,
                   'learning_rate': 0.01, 'loss': 'ls'}

plt.figure()

for label, color, setting in [('No shrinkage', 'orange',
                               {'n_estimators': 1000}),
                              ('n_estimators: 100', 'turquoise',
                               {'n_estimators': 100})]:
                              
#                               ('n_estimators: 250', 'max_depth: 4', 'blue',
#                                {'n_estimators': 250, 'max_depth': 4}),
#                               ('n_estimators: 500', 'max_depth: 4', 'gray',
#                                {'n_estimators': 500, 'max_depth': 4}),
#                               ('n_estimators: 750', 'max_depth: 4', 'magenta',
#                                {'n_estimators': 750, 'max_depth': 4})]:
    
# for label, color, setting in [('No shrinkage', 'orange',
#                                {'n_estimators': 1000, 'max_depth': 4}),
#                               ('n_estimators: 100', 'max_depth: 4', 'turquoise',
#                                {'n_estimators': 100, 'max_depth': 4}),
#                               ('n_estimators: 250', 'max_depth: 4', 'blue',
#                                {'n_estimators': 250, 'max_depth': 4}),
#                               ('n_estimators: 500', 'max_depth: 4', 'gray',
#                                {'n_estimators': 500, 'max_depth': 4}),
#                               ('n_estimators: 750', 'max_depth: 4', 'magenta',
#                                {'n_estimators': 750, 'max_depth': 4})]:
    
    params = dict(original_params)
    params.update(setting)

    #clf = ensemble.GradientBoostingClassifier(**params)
    clf.fit(X_train, y_train)

    # compute test set deviance
    test_deviance = np.zeros((params['n_estimators'],), dtype=np.float64)

    for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
        # clf.loss_ assumes that y_test[i] in {0, 1}
        test_deviance[i] = clf.loss_(y_test, y_pred)

    plt.plot((np.arange(test_deviance.shape[0]) + 1)[::5], test_deviance[::5],
            '-', color=color, label=label)

plt.legend(loc='upper left')
plt.xlabel('Boosting Iterations')
plt.ylabel('Test Set Deviance')

plt.show()


########################################################################


# In[58]:

X_train.shape,y_train.shape,y_test.shape,y_pred.shape


# In[115]:



plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Prediction')

plt.plot(y_test, y_pred, 
         label='Prediction Accuracy')
plt.legend(loc='upper right')
#plt.xlabel('Actual')
#plt.ylabel('Predicted')

plt.show()

# Release memory.
plt.clf()
plt.close()


# In[379]:

len(test_score)


# In[372]:

len(y_test)


# In[348]:

max_count = 0
for i, y_pred in enumerate(clf.staged_predict(X_test)):
    if i > max_count:
        max_count = i
print(max_count)


# In[253]:

(diamonds_target_use)


# In[117]:

#print(__doc__)

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target, random_state=13)
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]


params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf.predict(X_test))
print("MSE: %.4f" % mse)



# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_predict(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, boston.feature_names[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

plt.show()




# Release memory.
plt.clf()
plt.close()


# In[73]:

boston.data.shape


# In[188]:

boston.feature_names


# In[411]:

#How it should work

########################################################################
#print(__doc__)

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn import datasets


X, y = datasets.make_hastie_10_2(n_samples=12000, random_state=1)
X = X.astype(np.float32)

# map labels from {-1, 1} to {0, 1}
labels, y = np.unique(y, return_inverse=True)

X_train, X_test = X[:2000], X[2000:]
y_train, y_test = y[:2000], y[2000:]

original_params = {'n_estimators': 1000, 'max_leaf_nodes': 4, 'max_depth': None, 'random_state': 2,
                   'min_samples_split': 5}

plt.figure()

for label, color, setting in [('No shrinkage', 'orange',
                               {'learning_rate': 1.0, 'subsample': 1.0}),
                              ('learning_rate=0.1', 'turquoise',
                               {'learning_rate': 0.1, 'subsample': 1.0}),
                              ('subsample=0.5', 'blue',
                               {'learning_rate': 1.0, 'subsample': 0.5}),
                              ('learning_rate=0.1, subsample=0.5', 'gray',
                               {'learning_rate': 0.1, 'subsample': 0.5}),
                              ('learning_rate=0.1, max_features=2', 'magenta',
                               {'learning_rate': 0.1, 'max_features': 2})]:
    params = dict(original_params)
    params.update(setting)

    clf = ensemble.GradientBoostingClassifier(**params)
    clf.fit(X_train, y_train)

    # compute test set deviance
    test_deviance = np.zeros((params['n_estimators'],), dtype=np.float64)

    for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
        # clf.loss_ assumes that y_test[i] in {0, 1}
        test_deviance[i] = clf.loss_(y_test, y_pred)

    plt.plot((np.arange(test_deviance.shape[0]) + 1)[::5], test_deviance[::5],
            '-', color=color, label=label)

plt.legend(loc='upper left')
plt.xlabel('Boosting Iterations')
plt.ylabel('Test Set Deviance')

plt.show()


########################################################################


# In[243]:

print(diamonds_target_use)


# In[143]:

diamonds_norm = pd.concat([diamonds_train, diamonds_test], ignore_index=True)
diamonds_norm.head()
diamonds_norm.columns[11]


# In[ ]:




# In[399]:

for i, y_pred in enumerate(clf.staged_predict(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)


# In[401]:

str(y_test)


# In[97]:

str(diamonds_norm_matrix)


# In[59]:

train_y.describe()


# In[57]:

(train_y.info())

print(np.mean((predictions - test_y) ** 2)))


# In[60]:

model_output.head()


# # Feature Selection

# # Recursive feature elimination

# In[61]:

#Recursive feature elimination

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

names = list(train_X)
 
#use linear regression as the model
lr = LinearRegression()
#rank all features, i.e continue the elimination until the last one
rfe = RFE(lr, n_features_to_select=1)
rfe.fit(train_X,train_y)
 
print "Features sorted by their rank:"
print sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names))

# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)


# In[239]:

# for 0.18 version or newer, use:
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
# for pre-0.18 versions of scikit, use:
#from sklearn.cross_validation import cross_val_score

#X = # Some features
#y = # Some classes

clf = Lasso()
scores = cross_val_score(clf, train_X, train_y, cv=10)
print(scores)


# In[183]:

train_X.shape,train_y.shape


# In[ ]:




# In[74]:

#citation
#Davide Albanese, Michele Filosi, Roberto Visintainer, Samantha Riccadonna, Giuseppe Jurman and Cesare Furlanello. minerva and minepy: a C engine for the MINE suite and its R, Python and MATLAB wrappers. 
#Bioinformatics (2013) 29(3): 407-408 first published online December 14, 2012 doi:10.1093/bioinformatics/bts707.

from sklearn.linear_model import (LinearRegression, Ridge, 
                                  Lasso, RandomizedLasso)
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestRegressor
import numpy as np
#from minepy import MINE



ranks = {}

names = np.array(list(train_X))


def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks ))

X_train = np.array(train_X)
y_train = np.array(train_y)

# X_test = np.array(X_test)
# y_test = np.array(y_test)

#Reshape out Array from (37741L, 1L)) to 37741L,
y_train = np.reshape(y_train, -1)
#y_test = np.reshape(y_test, -1)

print "Features sorted by their coefficient:"
print('\n'*2)

print('Linear Regression')
print('\n'*2)
clf = LinearRegression(normalize=True)
clf.fit(X_train, y_train)
print(clf.coef_)
ranks["Linear reg"] = rank_to_dict(np.abs(clf.coef_), names)

print('\n'*2)


print sorted(zip(map(lambda x: round(x, 4), clf.coef_), names))
print('\n'*2)

print('Ridge Regression')
print('\n'*2)
clf = None
clf = Ridge(alpha=1.0)
clf.fit(X_train, y_train)
ranks["Ridge"] = rank_to_dict(np.abs(clf.coef_), names)
print sorted(zip(map(lambda x: round(x, 4), clf.coef_), names))
print('\n'*2)

print('Lasso Regression')
print('\n'*2)
clf = None
clf = Lasso(alpha=.05)
clf.fit(X_train, y_train)
ranks["Lasso"] = rank_to_dict(np.abs(clf.coef_), names)
print sorted(zip(map(lambda x: round(x, 4), clf.coef_), names))
print('\n'*2)

print('RandomizedLasso Regression')
clf = RandomizedLasso(alpha=0.04)
clf.fit(X_train, y_train)
ranks["Stability"] = rank_to_dict(np.abs(clf.scores_), names)
print sorted(zip(map(lambda x: round(x, 4), clf.scores_), names))
print('\n'*2)

#stop the search when 5 features are left (they will get equal scores)
print('Recursive Feature Elimination')
lr = LinearRegression(normalize=True)
clf = RFE(lr, n_features_to_select=5)
clf.fit(X_train, y_train)
ranks["RFE"] = rank_to_dict(map(float, clf.ranking_), names, order=-1)
print sorted(zip(map(lambda x: round(x, 4), clf.ranking_), names))
print('\n'*2) 

print('Random Forest Regressor')
clf = RandomForestRegressor()
clf.fit(X_train, y_train)
ranks["RF"] = rank_to_dict(clf.feature_importances_, names)
print sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), names))
print('\n'*2)

print('Correlation')
f, pval  = f_regression(X_train, y_train, center=True)
ranks["Corr."] = rank_to_dict(f, names)
print sorted(zip(map(lambda x: round(x, 4), f), names))
print('\n'*2)

print(ranks["Corr."])
print(ranks["RF"])



r = {}
for name in names:
    r[name] = round(np.mean([ranks[method][name] 
                             for method in ranks.keys()]), 2)

methods = sorted(ranks.keys())
ranks["Mean"] = r
methods.append("Mean")
 
print "\t%s" % "\t".join(methods)
for name in names:
    print "%s\t%s" % (name, "\t".join(map(str, 
                         [ranks[method][name] for method in methods])))


# In[62]:

clf = None
clf = Ridge(alpha=1.0)
clf.fit(X_train, y_train)
ranks["Ridge"] = rank_to_dict(np.abs(clf.coef_), names)
print sorted(zip(map(lambda x: round(x, 4), clf.coef_), names))
print('\n'*2)
print(sorted(zip(map(lambda x: round(x, 4), clf.coef_), names))[::-1])


# In[63]:

ranks[]


# In[65]:

ranks["Ridge"],ranks, dict(zip(names, ranks))#[::-1]


# In[64]:

#ranks["Ridge"],ranks, dict(zip(names, ranks))#[::-1]

ranks["Ridge"]


# In[197]:

minmax = MinMaxScaler()
order=1
my_val = minmax.fit_transform(order*np.array([ranks["Ridge"]]).T).T[0]


# In[199]:

names


# ### Recursive feature elimination

# In[66]:


from sklearn.linear_model import RandomizedLasso
from sklearn.datasets import load_boston

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
 
boston = load_boston()
X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]
 
#use linear regression as the model
lr = LinearRegression()
#rank all features, i.e continue the elimination until the last one
rfe = RFE(lr, n_features_to_select=1)
rfe.fit(X,Y)
 
print "Features sorted by their rank:"
print sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names))




# In[41]:

y_train.shape


# In[98]:

from sklearn.datasets import load_boston
from sklearn.linear_model import (LinearRegression, Ridge, 
                                  Lasso, RandomizedLasso)
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
#from minepy import MINE
 
np.random.seed(0)
 
size = 750
X = np.random.uniform(0, 1, (size, 14))
 
#"Friedamn #1” regression problem
Y = (10 * np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2] - .5)**2 +
     10*X[:,3] + 5*X[:,4] + np.random.normal(0,1))
#Add 3 additional correlated variables (correlated with X1-X3)
X[:,10:] = X[:,:4] + np.random.normal(0, .025, (size,4))
 
names = ["x%s" % i for i in range(1,15)]
 
ranks = {}
 
def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks ))
 
lr = LinearRegression(normalize=True)
lr.fit(X, Y)
ranks["LR"] = rank_to_dict(np.abs(lr.coef_), names) #Linear regression
 
ridge = Ridge(alpha=7)
ridge.fit(X, Y)
ranks["Ridge"] = rank_to_dict(np.abs(ridge.coef_), names) #Ridge
 

lasso = Lasso(alpha=.05)
lasso.fit(X, Y)
ranks["LS"] = rank_to_dict(np.abs(lasso.coef_), names)
 
lasso = Lasso(alpha=.0001)
lasso.fit(X, Y)
ranks["LS.0001"] = rank_to_dict(np.abs(lasso.coef_), names)#alpha 0.0001
    

    
rlasso = RandomizedLasso(alpha=0.04)
rlasso.fit(X, Y)
ranks["LSta"] = rank_to_dict(np.abs(rlasso.scores_), names) #Stability
 
#stop the search when 5 features are left (they will get equal scores)
rfe = RFE(lr, n_features_to_select=5)
rfe.fit(X,Y)
ranks["RFE"] = rank_to_dict(map(float, rfe.ranking_), names, order=-1)
 
rf = RandomForestRegressor()
rf.fit(X,Y)
ranks["RF"] = rank_to_dict(rf.feature_importances_, names)
 

f, pval  = f_regression(X, Y, center=True)
ranks["Corr."] = rank_to_dict(f, names)
 
    

    
# mine = MINE()
# mic_scores = []
# for i in range(X.shape[1]):
#     mine.compute_score(X[:,i], Y)
#     m = mine.mic()
#     mic_scores.append(m)
    
    
    
#ranks["MIC"] = rank_to_dict(mic_scores, names) 
 
r = {}
for name in names:
    r[name] = round(np.mean([ranks[method][name] 
                             for method in ranks.keys()]), 2)

methods = sorted(ranks.keys())
ranks["Mean"] = r
methods.append("Mean")
 
print "\t%s" % "\t".join(methods)
for name in names:
    print "%s\t%s" % (name, "\t".join(map(str, 
                         [ranks[method][name] for method in methods])))


# # Build Models based on RFE 

# In[138]:

train_full.head()#.iloc[:,[0,1,2,3,4,5,7,8,9]]



# In[132]:

train_X.iloc[:,[5,2,3,4]].head()


# In[79]:

from sklearn.linear_model import (LinearRegression, Ridge, 
                                  Lasso, RandomizedLasso)
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler

#from sklearn.ensemble import RandomForestRegressor
import numpy as np

from sklearn import metrics
from sklearn.metrics import r2_score

    

#List of combinations
comb = []

#Dictionary to store the MAE for all algorithms 
mae_list = []

output_columns = ['Model','r^2 on test data', 'Model Score on Training Data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']
model_output = pd.DataFrame(columns=output_columns)


#GradientBoostRegressor
#['y' 'clarity' 'z' 'carat' 'color' 'x' 'cut' 'depth' 'table']

#Recursive Feature Elimination top 5 predictors
#[(1.0, 'carat'), (1.0, 'clarity'), (1.0, 'x'), (1.0, 'y'), (1.0, 'z'), (2.0, 'color'), (3.0, 'depth'), (4.0, 'cut'), (5.0, 'table')]

#Ridge
# {'carat': 1.0, 
#  'x': 0.49, 
#  'y': 0.57, 
#  'z': 0.42}
#  'clarity': 0.06,
#  'color': 0.06,
#  'cut': 0.0,
#  'depth': 0.01,
#  'table': 0.0,


ranks = {}

names = np.array(list(train_X))


def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks ))


def build_model(X_to_use,y_to_use,X_to_test,y_to_test,tag):
    print "Features sorted by their coefficient:"
    print('\n'*2)

    print('Linear Regression', tag)
    print('\n'*2)
    model = LinearRegression(normalize=True)
    model.fit(X_to_use, y_to_use)
    print(model.coef_)
    #ranks["Linear reg"] = rank_to_dict(np.abs(clf.coef_), names)
    print('Training score', model.score(X_to_use, y_to_use))
    print('Training score','{0:.3f}'.format(model.score(X_to_test, y_to_test)))
    print('\n'*2)

    pred_test = model.fit(X_to_use, y_to_use).predict(X_to_test)
    predictions = pred_test
    r2_score_linear = r2_score(y_to_test, pred_test)


    model_r2_score = r2_score_linear

    #model_analysis(linear,tag,model_r2_score)

    mae = metrics.mean_absolute_error(y_to_test, pred_test)
    mse = metrics.mean_squared_error(y_to_test, pred_test)

    pred_len=int(len(predictions))

    output_columns = ['Model','r^2 on test data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']

    model_output.loc[model_output.shape[0]] = (tag,'{0:.6f}'.format(model_r2_score),'{0:.6f}'.format(model.score(X_to_test, y_to_test)),
                        '{0:.6f}'.format(model.score(X_to_use, y_to_use)),np.sqrt(metrics.mean_squared_error(y_to_test, pred_test)),(mae),mse)
    
   
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    my_title=('Prediction for Model: %s'%(model))
    plt.title(my_title +'\n')
    #plt.title('Prediction')
    plt.scatter(test_y, pred_test, 
         label='Prediction Accuracy')
    plt.legend(loc='upper left')
    #plt.xlabel('Actual')
    #plt.ylabel('Predicted')

    plt.show()

    # Release memory.
    plt.clf()
    plt.close()
    
    #Result obtained after running the algo. Comment the below two lines if you want to run the algo
    mae_list.append(mae)
    comb.append(tag)  

    
    
    

# Create linear regression object
print('Linear Regression -minimum')
tag = 'Linear Model lprice ~ y '

# train_X=diamonds_train[['carat','cut','clarity','x','y','z']]# taking the training data features
# train_y=diamonds_train[['lprice']] # output of our training data
# test_X =diamonds_test[['carat','cut','clarity','x','y','z']] # taking test data features
# test_y=diamonds_test[['lprice']]  #output value of test data


X_build = np.array(train_X.iloc[:,[5]])
y_build = np.array(train_y)
X_build_test = np.array(test_X.iloc[:,[5]])
y_build_test = np.array(test_y)

# # X_test = np.array(X_test)
# # y_test = np.array(y_test)

# #Reshape out Array from (37741L, 1L)) to 37741L,
# y_build = np.reshape(y_build, -1)
# #y_test = np.reshape(y_test, -1)

build_model(X_build,y_build,X_build_test,y_build_test,tag)

print('Linear Regression all')
tag = 'Linear Model lprice ~ carat + cut + clarity + color + x + y + z + depth + table '
X_build = np.array(train_X)
y_build = np.array(train_y)
X_build_test = np.array(test_X)

build_model(X_build,y_build,X_build_test,y_build_test,tag)


tag = 'Linear Model lprice ~ y + clarity '
X_build = np.array(train_X.iloc[:,[5,2]])
y_build = np.array(train_y)
X_build_test = np.array(test_X.iloc[:,[5,2]])

build_model(X_build,y_build,X_build_test,y_build_test,tag)


tag = 'Linear Model lprice ~ y + clarity + color '
X_build = np.array(train_X.iloc[:,[5,2,3]])
y_build = np.array(train_y)
X_build_test = np.array(test_X.iloc[:,[5,2,3]])

build_model(X_build,y_build,X_build_test,y_build_test,tag)

tag = 'Linear Model lprice ~ y + clarity + color + carat'
X_build = np.array(train_X.iloc[:,[5,2,3,0]])
y_build = np.array(train_y)
X_build_test = np.array(test_X.iloc[:,[5,2,3,0]])

build_model(X_build,y_build,X_build_test,y_build_test,tag)

tag = 'Linear Model lprice ~ y + clarity + color + x'
X_build = np.array(train_X.iloc[:,[5,2,3,4]])
y_build = np.array(train_y)
X_build_test = np.array(test_X.iloc[:,[5,2,3,4]])

build_model(X_build,y_build,X_build_test,y_build_test,tag)

model_output



# In[125]:

clf = LinearRegression(normalize=True)
clf.fit(X_train, y_train)
print(clf.coef_)


# In[48]:

np.array(train_X['y']).shape,np.array(train_y).shape


# In[78]:

#Build Linear Models Using OLS
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm



#GradientBoostRegressor
#['y' 'clarity' 'z' 'carat' 'color' 'x' 'cut' 'depth' 'table']

#Recursive Feature Elimination top 5 predictors
#[(1.0, 'carat'), (1.0, 'clarity'), (1.0, 'x'), (1.0, 'y'), (1.0, 'z'), (2.0, 'color'), (3.0, 'depth'), (4.0, 'cut'), (5.0, 'table')]

#Ridge
# {'carat': 1.0, 
#  'x': 0.49, 
#  'y': 0.57, 
#  'z': 0.42}
#  'clarity': 0.06,
#  'color': 0.06,
#  'cut': 0.0,
#  'depth': 0.01,
#  'table': 0.0,


def run_ols_build(label,features):
    df = diamonds
    result = sm.ols(formula=label + "~" + build_features, data=df).fit()
    print result.params
    print result.summary()
    print("\n"*2)

#y, X = dmatrices(label + '~' + features,

#names = list(train_X)

build_features = "+".join(train_X.columns  - ['carat','cut','clarity','x','z','color','depth','table'])
#Pick out label                   
label = "".join(train_y.columns)
run_ols_build(label,features)


build_features = "+".join(train_X.columns  - ['carat','cut','x','z','color','depth','table'])

run_ols_build(label,features)    


build_features = "+".join(train_X.columns  - ['carat','cut','clarity','x','z','color','depth','table'])

run_ols_build(label,features)   


build_features = "+".join(train_X.columns  - ['carat','cut','x','z','depth','table'])

run_ols_build(label,features)   

#Some benefit to add carat
build_features = "+".join(train_X.columns  - ['cut','x','z','depth','table'])

run_ols_build(label,features)   


#No benefit to add cut
build_features = "+".join(train_X.columns  - ['carat','x','z','depth','table'])

run_ols_build(label,features) 

#Try drop y and use carat - not as good at explaining data
build_features = "+".join(train_X.columns  - ['y','cut','x','z','depth','table'])
#Pick out label                   
label = "".join(train_y.columns)
run_ols_build(label,features)

#Use our clarity,colour and our 3-axes
build_features = "+".join(train_X.columns  - ['carat','cut','depth','table'])
#Pick out label                   
label = "".join(train_y.columns)
run_ols_build(label,features)


# df = diamonds
# result = sm.ols(formula=label + "~" + build_features, data=df).fit()

#rank all features, i.e continue the elimination until the last one
#rfe = RFE(lr, n_features_to_select=1)
#rfe.fit(train_X,train_y)
 
#print "Features sorted by their rank:"
#print sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names))



# In[77]:


#Build Linear Models Using OLS
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm



#GradientBoostRegressor
#['y' 'clarity' 'z' 'carat' 'color' 'x' 'cut' 'depth' 'table']

#Recursive Feature Elimination top 5 predictors
#[(1.0, 'carat'), (1.0, 'clarity'), (1.0, 'x'), (1.0, 'y'), (1.0, 'z'), (2.0, 'color'), (3.0, 'depth'), (4.0, 'cut'), (5.0, 'table')]

#Ridge
# {'carat': 1.0, 
#  'x': 0.49, 
#  'y': 0.57, 
#  'z': 0.42}
#  'clarity': 0.06,
#  'color': 0.06,
#  'cut': 0.0,
#  'depth': 0.01,
#  'table': 0.0,


def run_ols_build(label,features):
    df = diamonds
    result = sm.ols(formula=label + "~" + build_features, data=df).fit()
    print (label + "~" + build_features)
    print("\n"*1)
    print result.params
    print result.summary()
    print("\n"*2)

#y, X = dmatrices(label + '~' + features,

#names = list(train_X)
#only y
train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
build_features = "+".join(train_X.columns  - ['carat','cut','clarity','x','z','color','depth','table'])
#Pick out label                   
label = "".join(train_y.columns)
run_ols_build(label,build_features)

#All predictors
train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
build_features = "+".join(train_X.columns)
#Pick out label                   
label = "".join(train_y.columns)
run_ols_build(label,build_features)

#All predictors
train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
build_features = "+".join(train_X.columns  - ['carat','x','z'])
#Pick out label                   
label = "".join(train_y.columns)
run_ols_build(label,build_features)


train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
build_features = "+".join(train_X.columns  - ['carat','cut','x','z','color','depth','table'])
run_ols_build(label,build_features)    

train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
build_features = "+".join(train_X.columns  - ['carat','cut','clarity','x','z','color','depth','table'])
run_ols_build(label,build_features)   

train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features

build_features = "+".join(train_X.columns  - ['carat','cut','x','z','depth','table'])
run_ols_build(label,build_features)   

#Some benefit to add carat
train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features

build_features = "+".join(train_X.columns  - ['cut','x','z','depth','table'])
run_ols_build(label,build_features)   


#No benefit to add cut
train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
build_features = "+".join(train_X.columns  - ['carat','x','z','depth','table'])
run_ols_build(label,build_features) 

#Try drop y and use carat - not as good at explaining data
train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
build_features = "+".join(train_X.columns  - ['y','cut','x','z','depth','table'])
#Pick out label                   
label = "".join(train_y.columns)
run_ols_build(label,build_features)

#Use our clarity,colour and our 3-axes
train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
build_features = "+".join(train_X.columns  - ['carat','cut','depth','table'])
#Pick out label                   
label = "".join(train_y.columns)
run_ols_build(label,build_features)


# df = diamonds
# result = sm.ols(formula=label + "~" + build_features, data=df).fit()

#rank all features, i.e continue the elimination until the last one
#rfe = RFE(lr, n_features_to_select=1)
#rfe.fit(train_X,train_y)
 
#print "Features sorted by their rank:"
#print sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names))

train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features


# In[ ]:

#Check all pedictors and subset


# In[47]:

train_X.columns  - ['carat','x','z',]


# In[27]:

train_y.info()


# In[95]:

#http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_and_elasticnet.html#sphx-glr-auto-examples-linear-model-plot-lasso-and-elasticnet-py
#Working Lasso Example 

def model_analysis (model,tag,model_r2_score):
#     print('Model Score on Training Data')

#     print(model.score(train_X, train_y))

#     print('Coefficients: \n', model.coef_)
#     print('Intercept: \n', model.intercept_)

#     # The mean squared error
#     print("Mean squared error: %.2f"
#       % np.mean((predictions - test_y) ** 2))

#     # Explained variance score: 1 is perfect prediction
#     print('Variance score: %.2f' % model.score(test_X, test_y))

#     print('\n'*1)
#     #Root mean Squared error
#     print ('\nRoot Mean Squared Error')

#     print np.sqrt(metrics.mean_squared_error(test_y,predictions))


    mae = metrics.mean_absolute_error(test_y, predictions)
    mse = metrics.mean_squared_error(test_y, predictions)

#     print('\nMean Absolute Error')
#     print(mae)

#     print('\nMean Squared Error')
#     print(mse)

    pred_len=int(len(predictions))
#     print(pred_len)

    output_columns = ['Model','r^2 on test data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']

    #df = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB')) 
    #df.loc[len(df)]=['8/19/2014','Jun','Fly','98765'] 
    model_output.loc[model_output.shape[0]] = (tag,'{0:.6f}'.format(model_r2_score),'{0:.6f}'.format(model.score(test_X, test_y)),
                        model.score(train_X, test_y),np.sqrt(metrics.mean_squared_error(test_y,predictions)),(mae),mse)
    #'{0:.6f}'.format(model.score(X_to_use, y_to_use)),np.sqrt(metrics.mean_squared_error(y_to_test, pred_test)),(mae),mse)
    
    #model_output.append(c(model,model_r2_score,np.mean((predictions - test_y) ** 2),
                        #model.score(test_X, test_y),np.sqrt(metrics.mean_squared_error(test_y,predictions)),mae,mse))
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    my_title=('Prediction for Model: %s'%(model))
    plt.title(my_title +'\n')
    #plt.title('Prediction')
    plt.scatter(test_y, pred_test, 
         label='Prediction Accuracy')
    plt.legend(loc='upper left')
    #plt.xlabel('Actual')
    #plt.ylabel('Predicted')

    plt.show()

    # Release memory.
    plt.clf()
    plt.close()
    
    #Result obtained after running the algo. Comment the below two lines if you want to run the algo
    mae_list.append(mae)
    comb.append(tag)  



train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
train_y=diamonds_train[['lprice']] # output of our training data
test_X =diamonds_test[['carat','cut','clarity','color','x','y','z','depth','table']] # taking test data features
test_y=diamonds_test[['lprice']]  #output value of test data

from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
    

#List of combinations
comb = []

#Dictionary to store the MAE for all algorithms 
mae_list = []

output_columns = ['Model','r^2 on test data', 'Model Score on Training Data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']
model_output = pd.DataFrame(columns=output_columns)

# print(model_output.head())


alpha = 0.1
linear = LinearRegression()
tag = 'linear'

pred_test = linear.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_linear = r2_score(test_y, pred_test)

#print(linear.fit)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_linear)

model_r2_score = r2_score_linear

model_analysis(linear,tag,model_r2_score)

print('\n'*1)


#Ridge
alpha = 1.0
ridge = Ridge(alpha=alpha)
tag = 'ridge'

pred_test = ridge.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_ridge = r2_score(test_y, pred_test)

# print(ridge)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_ridge)

model_r2_score = r2_score_ridge

model_analysis(ridge,tag,model_r2_score)

print('\n'*1)


#Lasso

alpha = 0.1
lasso = Lasso(alpha=alpha)
tag = 'lasso'

pred_test = lasso.fit(train_X, train_y).predict(test_X)

## lets see what the coefficients are
print dict(zip(train_X.columns, lasso.coef_))

predictions = pred_test
r2_score_lasso = r2_score(test_y, pred_test)


# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

model_r2_score = r2_score_lasso

model_analysis(lasso,tag,model_r2_score)

print('\n'*1)

# alpha = 1.0
# lasso = Lasso(alpha=alpha)

# pred_test = lasso.fit(train_X, train_y).predict(test_X)
# r2_score_lasso = r2_score(test_y, pred_test)
# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

# print('\n'*1)

#ElasticNet

enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
tag = 'enet'

#y_pred_enet = enet.fit(train_X, train_y).predict(test_X)

pred_test = enet.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_enet = r2_score(test_y, pred_test)

# print(enet)
# print("r^2 on test data : %f" % r2_score_enet)

model_r2_score = r2_score_enet

model_analysis(enet,tag,model_r2_score)

#Unlike most other scores, R^2 score may be negative (it need not actually be the square of a quantity R).

print('\n'*1)
print ('Model formula : lprice ~ carat + cut + clarity + color + x + y + z + depth + table')

model_output.head()


# In[76]:

#http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_and_elasticnet.html#sphx-glr-auto-examples-linear-model-plot-lasso-and-elasticnet-py
#Working Lasso Example 

def model_analysis (model,tag,model_r2_score):
#     print('Model Score on Training Data')

#     print(model.score(train_X, train_y))

#     print('Coefficients: \n', model.coef_)
#     print('Intercept: \n', model.intercept_)

#     # The mean squared error
#     print("Mean squared error: %.2f"
#       % np.mean((predictions - test_y) ** 2))

#     # Explained variance score: 1 is perfect prediction
#     print('Variance score: %.2f' % model.score(test_X, test_y))

#     print('\n'*1)
#     #Root mean Squared error
#     print ('\nRoot Mean Squared Error')

#     print np.sqrt(metrics.mean_squared_error(test_y,predictions))


    mae = metrics.mean_absolute_error(test_y, predictions)
    mse = metrics.mean_squared_error(test_y, predictions)

#     print('\nMean Absolute Error')
#     print(mae)

#     print('\nMean Squared Error')
#     print(mse)

    pred_len=int(len(predictions))
#     print(pred_len)

    output_columns = ['Model','r^2 on test data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']

    #df = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB')) 
    #df.loc[len(df)]=['8/19/2014','Jun','Fly','98765'] 
    model_output.loc[model_output.shape[0]] = (tag,'{0:.6f}'.format(model_r2_score),'{0:.6f}'.format(model.score(test_X, test_y)),
                        '{0:.6f}'.format(model.score(train_X, train_y)),np.sqrt(metrics.mean_squared_error(test_y,predictions)),(mae),mse)
    #'{0:.6f}'.format(model.score(X_to_use, y_to_use)),np.sqrt(metrics.mean_squared_error(y_to_test, pred_test)),(mae),mse)
    
    #model_output.append(c(model,model_r2_score,np.mean((predictions - test_y) ** 2),
                        #model.score(test_X, test_y),np.sqrt(metrics.mean_squared_error(test_y,predictions)),mae,mse))
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    my_title=('Prediction for Model: %s'%(model))
    plt.title(my_title +'\n')
    #plt.title('Prediction')
    plt.scatter(test_y, pred_test, 
         label='Prediction Accuracy')
    plt.legend(loc='upper left')
    #plt.xlabel('Actual')
    #plt.ylabel('Predicted')

    plt.show()

    # Release memory.
    plt.clf()
    plt.close()
    
    #Result obtained after running the algo. Comment the below two lines if you want to run the algo
    mae_list.append(mae)
    comb.append(tag)  

def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks ))    
    

train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
train_y=diamonds_train[['lprice']] # output of our training data
test_X =diamonds_test[['carat','cut','clarity','color','x','y','z','depth','table']] # taking test data features
test_y=diamonds_test[['lprice']]  #output value of test data

from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

from sklearn.preprocessing import MinMaxScaler

#from sklearn.ensemble import RandomForestRegressor
import numpy as np



ranks = {}

names = np.array(list(train_X))

#List of combinations
comb = []

#Dictionary to store the MAE for all algorithms 
mae_list = []

output_columns = ['Model','r^2 on test data', 'Model Score on Training Data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']
model_output = pd.DataFrame(columns=output_columns)

# print(model_output.head())


alpha = 0.1
linear = LinearRegression()
tag = 'linear'

pred_test = linear.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_linear = r2_score(test_y, pred_test)

#print(linear.fit)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_linear)

model_r2_score = r2_score_linear

model_analysis(linear,tag,model_r2_score)

print('\n'*1)


#Ridge
alpha = 1.0
ridge = Ridge(alpha=alpha)
tag = 'ridge'

pred_test = ridge.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_ridge = r2_score(test_y, pred_test)

# print(ridge)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_ridge)

model_r2_score = r2_score_ridge

model_analysis(ridge,tag,model_r2_score)

print('\n'*1)


#Lasso

alpha = 0.1
lasso = Lasso(alpha=alpha)
tag = 'lasso alpha = 0.1'

pred_test = lasso.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_lasso = r2_score(test_y, pred_test)

#Let's look for variable selection
print dict(zip(train_X.columns, lasso.coef_))

#################################################

#print "Features sorted by their coefficient:"
#print('\n'*1)

ranks["lasso alpha = 0.1"] = rank_to_dict(np.abs(lasso.coef_), names)
#print (ranks["lasso alpha = 0.1"])
#print('\n'*2)

##################################################

# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

model_r2_score = r2_score_lasso

model_analysis(lasso,tag,model_r2_score)

print('\n'*1)



#Lasso 0.0001

alpha = 0.0001
lasso = Lasso(alpha=alpha)
tag = 'lasso alpha = 0.0001'

pred_test = lasso.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_lasso = r2_score(test_y, pred_test)

#Let's look for variable selection
print dict(zip(train_X.columns, lasso.coef_))
#################################################

#print "Features sorted by their coefficient:"
#print('\n'*1)

ranks["lasso alpha = 0.0001"] = rank_to_dict(np.abs(lasso.coef_), names)
#print (ranks["lasso alpha = 0.0001"])
#print('\n'*2)

##################################################

# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

model_r2_score = r2_score_lasso

model_analysis(lasso,tag,model_r2_score)

print('\n'*1)




#Lasso 0.00047

alpha = 0.00047
lasso = Lasso(alpha=alpha)
tag = 'lasso alpha = 0.00047'

pred_test = lasso.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_lasso = r2_score(test_y, pred_test)

#Let's look for variable selection
print dict(zip(train_X.columns, lasso.coef_))
#################################################
#print "Features sorted by their coefficient:"
#print('\n'*1)

ranks["lasso alpha = 0.00047"] = rank_to_dict(np.abs(lasso.coef_), names)
#print (ranks["lasso alpha = 0.00047"])
#print('\n'*2)
##################################################
# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

model_r2_score = r2_score_lasso

model_analysis(lasso,tag,model_r2_score)

print('\n'*1)



#ElasticNet
alpha = 0.1
enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
tag = 'enet alpha = 0.1'

#y_pred_enet = enet.fit(train_X, train_y).predict(test_X)

pred_test = enet.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_enet = r2_score(test_y, pred_test)

# print(enet)
# print("r^2 on test data : %f" % r2_score_enet)

model_r2_score = r2_score_enet

model_analysis(enet,tag,model_r2_score)

#ElasticNet
alpha = 0.0001
enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
tag = 'enet alpha = 0.0001'

#y_pred_enet = enet.fit(train_X, train_y).predict(test_X)
pred_test= enet.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_enet = r2_score(test_y, pred_test)

# print(enet)
# print("r^2 on test data : %f" % r2_score_enet)

model_r2_score = r2_score_enet

model_analysis(enet,tag,model_r2_score)

#ElasticNet
alpha = 0.00094
enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
tag = 'enet alpha = 0.00094'

#y_pred_enet = enet.fit(train_X, train_y).predict(test_X)
pred_test= enet.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_enet = r2_score(test_y, pred_test)

# print(enet)
# print("r^2 on test data : %f" % r2_score_enet)

model_r2_score = r2_score_enet

model_analysis(enet,tag,model_r2_score)

#Unlike most other scores, R^2 score may be negative (it need not actually be the square of a quantity R).

####Lasso feature Selection
print "Lasso features sorted by their coefficient:"
print('\n'*1)
print ("alpha = 0.1")
print(ranks["lasso alpha = 0.1"])
print ('alpha = 0.0001')
print(ranks["lasso alpha = 0.0001"])
print ('alpha = 0.00047')
print(ranks["lasso alpha = 0.00047"])

print('\n'*1)
print ('Model formula : lprice ~ carat + cut + clarity + color + x + y + z + depth + table')

model_output


# ### Lets see the prediction plots for price as well on all the predictors

# In[11]:

np.exp(pred_test)


# In[75]:

train_y.head()


# In[80]:

#http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_and_elasticnet.html#sphx-glr-auto-examples-linear-model-plot-lasso-and-elasticnet-py
#Working Lasso Example 

def model_analysis (model,tag,model_r2_score):
#     print('Model Score on Training Data')

#     print(model.score(train_X, train_y))

#     print('Coefficients: \n', model.coef_)
#     print('Intercept: \n', model.intercept_)

#     # The mean squared error
#     print("Mean squared error: %.2f"
#       % np.mean((predictions - test_y) ** 2))

#     # Explained variance score: 1 is perfect prediction
#     print('Variance score: %.2f' % model.score(test_X, test_y))

#     print('\n'*1)
#     #Root mean Squared error
#     print ('\nRoot Mean Squared Error')

#     print np.sqrt(metrics.mean_squared_error(test_y,predictions))

    mae = metrics.mean_absolute_error(test_y, predictions)
    mse = metrics.mean_squared_error(test_y, predictions)

#     print('\nMean Absolute Error')
#     print(mae)

#     print('\nMean Squared Error')
#     print(mse)

    pred_len=int(len(predictions))
#     print(pred_len)

    output_columns = ['Model','r^2 on test data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']

    #df = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB')) 
    #df.loc[len(df)]=['8/19/2014','Jun','Fly','98765'] 
    model_output.loc[model_output.shape[0]] = (tag,'{0:.6f}'.format(model_r2_score),'{0:.6f}'.format(model.score(test_X, test_y)),
                        '{0:.6f}'.format(model.score(train_X, train_y)),np.sqrt(metrics.mean_squared_error(test_y,predictions)),(mae),mse)
    #'{0:.6f}'.format(model.score(X_to_use, y_to_use)),np.sqrt(metrics.mean_squared_error(y_to_test, pred_test)),(mae),mse)
    
    #model_output.append(c(model,model_r2_score,np.mean((predictions - test_y) ** 2),
                        #model.score(test_X, test_y),np.sqrt(metrics.mean_squared_error(test_y,predictions)),mae,mse))
    
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     my_title=('Prediction for Model: %s'%(model))
#     plt.title(my_title +'\n')
#     #plt.title('Prediction')
#     plt.scatter(test_y, pred_test, 
#          label='Prediction Accuracy')
#     plt.legend(loc='upper left')
#     #plt.xlabel('Actual')
#     #plt.ylabel('Predicted')

#     plt.show()

#     # Release memory.
#     plt.clf()
#     plt.close()

    #plt.figure(figsize=(12, 6))
    plt.figure(figsize=(6, 4))
    #plt.subplot(1, 2, 1)
    my_title=('Prediction for Model: %s'%(model))
    plt.title(my_title +'\n')
    #plt.title('Prediction')
    #plt.scatter(test_y, pred_test, 
         #label='Prediction Accuracy')
    plt.scatter(pred_test, test_y , 
         label='Prediction Accuracy')
    plt.legend(loc='upper left')
    #plt.xlabel('Actual')
    #plt.ylabel('Predicted')

    plt.show()
    
    plt.figure(figsize=(6, 4))
    #plt.subplot(1, 2, 2)
    my_title=('Price Prediction for Model: %s'%(model))
    plt.title(my_title +'\n')
    #plt.title('Prediction')
    #plt.scatter(np.exp(test_y), np.exp(pred_test), 
         #label='Prediction Accuracy')
    plt.scatter(np.exp(pred_test),np.exp(test_y), 
         label='Price Prediction')
    plt.legend(loc='lower right')
    
    plt.show()

    # Release memory.
    plt.clf()
    plt.close()
    
    #Result obtained after running the algo. Comment the below two lines if you want to run the algo
    mae_list.append(mae)
    comb.append(tag)  

def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks ))    
    

train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
train_y=diamonds_train[['lprice']] # output of our training data
test_X =diamonds_test[['carat','cut','clarity','color','x','y','z','depth','table']] # taking test data features
test_y=diamonds_test[['lprice']]  #output value of test data

from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

from sklearn.preprocessing import MinMaxScaler

#from sklearn.ensemble import RandomForestRegressor
import numpy as np



ranks = {}

names = np.array(list(train_X))

#List of combinations
comb = []

#Dictionary to store the MAE for all algorithms 
mae_list = []

output_columns = ['Model','r^2 on test data', 'Model Score on Training Data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']
model_output = pd.DataFrame(columns=output_columns)

# print(model_output.head())


alpha = 0.1
linear = LinearRegression()
tag = 'linear'

pred_test = linear.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_linear = r2_score(test_y, pred_test)


# print("HERE")
# print(np.exp(pred_test))
# print(train_X.head())

# print("HERE")

#print (pred_test)
#print(linear.fit)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_linear)

model_r2_score = r2_score_linear

model_analysis(linear,tag,model_r2_score)

print('\n'*1)


#Ridge
alpha = 1.0
ridge = Ridge(alpha=alpha)
tag = 'ridge'

pred_test = ridge.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_ridge = r2_score(test_y, pred_test)

# print(ridge)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_ridge)

model_r2_score = r2_score_ridge

model_analysis(ridge,tag,model_r2_score)

print('\n'*1)


#Lasso

alpha = 0.1
lasso = Lasso(alpha=alpha)
tag = 'lasso alpha = 0.1'

pred_test = lasso.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_lasso = r2_score(test_y, pred_test)

#Let's look for variable selection
print dict(zip(train_X.columns, lasso.coef_))

#################################################

#print "Features sorted by their coefficient:"
#print('\n'*1)

ranks["lasso alpha = 0.1"] = rank_to_dict(np.abs(lasso.coef_), names)
#print (ranks["lasso alpha = 0.1"])
#print('\n'*2)

##################################################

# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

model_r2_score = r2_score_lasso

model_analysis(lasso,tag,model_r2_score)

print('\n'*1)



#Lasso 0.0001

alpha = 0.0001
lasso = Lasso(alpha=alpha)
tag = 'lasso alpha = 0.0001'

pred_test = lasso.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_lasso = r2_score(test_y, pred_test)

#Let's look for variable selection
print dict(zip(train_X.columns, lasso.coef_))
#################################################

#print "Features sorted by their coefficient:"
#print('\n'*1)

ranks["lasso alpha = 0.0001"] = rank_to_dict(np.abs(lasso.coef_), names)
#print (ranks["lasso alpha = 0.0001"])
#print('\n'*2)

##################################################

# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

model_r2_score = r2_score_lasso

model_analysis(lasso,tag,model_r2_score)

print('\n'*1)




#Lasso 0.00047

alpha = 0.00047
lasso = Lasso(alpha=alpha)
tag = 'lasso alpha = 0.00047'

pred_test = lasso.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_lasso = r2_score(test_y, pred_test)

#Let's look for variable selection
print dict(zip(train_X.columns, lasso.coef_))
#################################################
#print "Features sorted by their coefficient:"
#print('\n'*1)

ranks["lasso alpha = 0.00047"] = rank_to_dict(np.abs(lasso.coef_), names)
#print (ranks["lasso alpha = 0.00047"])
#print('\n'*2)
##################################################
# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

model_r2_score = r2_score_lasso

model_analysis(lasso,tag,model_r2_score)

print('\n'*1)



#ElasticNet
alpha = 0.1
enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
tag = 'enet alpha = 0.1'

#y_pred_enet = enet.fit(train_X, train_y).predict(test_X)

pred_test = enet.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_enet = r2_score(test_y, pred_test)

# print(enet)
# print("r^2 on test data : %f" % r2_score_enet)

model_r2_score = r2_score_enet

model_analysis(enet,tag,model_r2_score)

#ElasticNet
alpha = 0.0001
enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
tag = 'enet alpha = 0.0001'

#y_pred_enet = enet.fit(train_X, train_y).predict(test_X)
pred_test= enet.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_enet = r2_score(test_y, pred_test)

# print(enet)
# print("r^2 on test data : %f" % r2_score_enet)

model_r2_score = r2_score_enet

model_analysis(enet,tag,model_r2_score)

#ElasticNet
alpha = 0.00094
enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
tag = 'enet alpha = 0.00094'

#y_pred_enet = enet.fit(train_X, train_y).predict(test_X)
pred_test= enet.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_enet = r2_score(test_y, pred_test)

# print(enet)
# print("r^2 on test data : %f" % r2_score_enet)

model_r2_score = r2_score_enet

model_analysis(enet,tag,model_r2_score)

#Unlike most other scores, R^2 score may be negative (it need not actually be the square of a quantity R).

####Lasso feature Selection
print "Lasso features sorted by their coefficient:"
print('\n'*1)
print ("alpha = 0.1")
print(ranks["lasso alpha = 0.1"])
print ('alpha = 0.0001')
print(ranks["lasso alpha = 0.0001"])
print ('alpha = 0.00047')
print(ranks["lasso alpha = 0.00047"])

print('\n'*1)
print ('Model formula : lprice ~ carat + cut + clarity + color + x + y + z + depth + table')

model_output


# In[74]:

#http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_and_elasticnet.html#sphx-glr-auto-examples-linear-model-plot-lasso-and-elasticnet-py
#Working Lasso Example 

def model_analysis (model,tag,model_r2_score):
#     print('Model Score on Training Data')

#     print(model.score(train_X, train_y))

#     print('Coefficients: \n', model.coef_)
#     print('Intercept: \n', model.intercept_)

#     # The mean squared error
#     print("Mean squared error: %.2f"
#       % np.mean((predictions - test_y) ** 2))

#     # Explained variance score: 1 is perfect prediction
#     print('Variance score: %.2f' % model.score(test_X, test_y))

#     print('\n'*1)
#     #Root mean Squared error
#     print ('\nRoot Mean Squared Error')

#     print np.sqrt(metrics.mean_squared_error(test_y,predictions))


    mae = metrics.mean_absolute_error(test_y, predictions)
    mse = metrics.mean_squared_error(test_y, predictions)

#     print('\nMean Absolute Error')
#     print(mae)

#     print('\nMean Squared Error')
#     print(mse)

    pred_len=int(len(predictions))
#     print(pred_len)

    output_columns = ['Model','r^2 on test data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']

    #df = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB')) 
    #df.loc[len(df)]=['8/19/2014','Jun','Fly','98765'] 
    model_output.loc[model_output.shape[0]] = (tag,'{0:.6f}'.format(model_r2_score),'{0:.6f}'.format(model.score(test_X, test_y)),
                        '{0:.6f}'.format(model.score(train_X, train_y)),np.sqrt(metrics.mean_squared_error(test_y,predictions)),(mae),mse)
    #'{0:.6f}'.format(model.score(X_to_use, y_to_use)),np.sqrt(metrics.mean_squared_error(y_to_test, pred_test)),(mae),mse)
    
    #model_output.append(c(model,model_r2_score,np.mean((predictions - test_y) ** 2),
                        #model.score(test_X, test_y),np.sqrt(metrics.mean_squared_error(test_y,predictions)),mae,mse))
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    my_title=('Prediction for Model: %s'%(model))
    plt.title(my_title +'\n')
    #plt.title('Prediction')
    plt.scatter(test_y, pred_test, 
         label='Prediction Accuracy')
    plt.legend(loc='upper left')
    #plt.xlabel('Actual')
    #plt.ylabel('Predicted')

    plt.show()

    # Release memory.
    plt.clf()
    plt.close()
    
    #Result obtained after running the algo. Comment the below two lines if you want to run the algo
    mae_list.append(mae)
    comb.append(tag)  



train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
train_y=diamonds_train[['lprice']] # output of our training data
test_X =diamonds_test[['carat','cut','clarity','color','x','y','z','depth','table']] # taking test data features
test_y=diamonds_test[['lprice']]  #output value of test data

from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
    

#List of combinations
comb = []

#Dictionary to store the MAE for all algorithms 
mae_list = []

output_columns = ['Model','r^2 on test data', 'Model Score on Training Data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']
model_output = pd.DataFrame(columns=output_columns)

# print(model_output.head())


alpha = 0.1
linear = LinearRegression()
tag = 'linear'

pred_test = linear.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_linear = r2_score(test_y, pred_test)

#print(linear.fit)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_linear)

model_r2_score = r2_score_linear

model_analysis(linear,tag,model_r2_score)

print('\n'*1)


#Ridge
alpha = 1.0
ridge = Ridge(alpha=alpha)
tag = 'ridge'

pred_test = ridge.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_ridge = r2_score(test_y, pred_test)

# print(ridge)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_ridge)

model_r2_score = r2_score_ridge

model_analysis(ridge,tag,model_r2_score)

print('\n'*1)


#Lasso

alpha = 0.1
lasso = Lasso(alpha=alpha)
tag = 'lasso alpha = 0.1'

pred_test = lasso.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_lasso = r2_score(test_y, pred_test)

# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

model_r2_score = r2_score_lasso

model_analysis(lasso,tag,model_r2_score)

print('\n'*1)

#Lasso 0.0001

alpha = 0.0001
lasso = Lasso(alpha=alpha)
tag = 'lasso alpha = 0.0001'

pred_test = lasso.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_lasso = r2_score(test_y, pred_test)

# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

model_r2_score = r2_score_lasso

model_analysis(lasso,tag,model_r2_score)

print('\n'*1)

#Lasso 0.0001

alpha = 0.00047
lasso = Lasso(alpha=alpha)
tag = 'lasso alpha = 0.00047'

pred_test = lasso.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_lasso = r2_score(test_y, pred_test)

# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

model_r2_score = r2_score_lasso

model_analysis(lasso,tag,model_r2_score)

print('\n'*1)


# alpha = 1.0
# lasso = Lasso(alpha=alpha)

# pred_test = lasso.fit(train_X, train_y).predict(test_X)
# r2_score_lasso = r2_score(test_y, pred_test)
# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

# print('\n'*1)

#ElasticNet
alpha = 0.1
enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
tag = 'enet alpha = 0.1'

#y_pred_enet = enet.fit(train_X, train_y).predict(test_X)

pred_test = enet.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_enet = r2_score(test_y, pred_test)

# print(enet)
# print("r^2 on test data : %f" % r2_score_enet)

model_r2_score = r2_score_enet

model_analysis(enet,tag,model_r2_score)

#ElasticNet
alpha = 0.0001
enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
tag = 'enet alpha = 0.0001'

#y_pred_enet = enet.fit(train_X, train_y).predict(test_X)
pred_test= enet.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_enet = r2_score(test_y, pred_test)

# print(enet)
# print("r^2 on test data : %f" % r2_score_enet)

model_r2_score = r2_score_enet

model_analysis(enet,tag,model_r2_score)

#ElasticNet
alpha = 0.00094
enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
tag = 'enet alpha = 0.00094'

#y_pred_enet = enet.fit(train_X, train_y).predict(test_X)
pred_test= enet.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_enet = r2_score(test_y, pred_test)

# print(enet)
# print("r^2 on test data : %f" % r2_score_enet)

model_r2_score = r2_score_enet

model_analysis(enet,tag,model_r2_score)

#Unlike most other scores, R^2 score may be negative (it need not actually be the square of a quantity R).

print('\n'*1)
print ('Model formula : lprice ~ carat + cut + clarity + color + x + y + z + depth + table')

model_output


# # Let's try this with only a few predictors  

# In[81]:

#http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_and_elasticnet.html#sphx-glr-auto-examples-linear-model-plot-lasso-and-elasticnet-py
#Working Lasso Example 

def model_analysis (model,tag,model_r2_score):


    mae = metrics.mean_absolute_error(test_y, predictions)
    mse = metrics.mean_squared_error(test_y, predictions)


    pred_len=int(len(predictions))

    output_columns = ['Model','r^2 on test data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']
 
    model_output.loc[model_output.shape[0]] = (tag,'{0:.6f}'.format(model_r2_score),'{0:.6f}'.format(model.score(test_X, test_y)),
                        '{0:.6f}'.format(model.score(train_X, train_y)),np.sqrt(metrics.mean_squared_error(test_y,predictions)),(mae),mse)
    
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    my_title=('Prediction for Model: %s'%(model))
    plt.title(my_title +'\n')
    #plt.title('Prediction')
    plt.scatter(test_y, pred_test, 
         label='Prediction Accuracy')
    plt.legend(loc='upper left')
    #plt.xlabel('Actual')
    #plt.ylabel('Predicted')

    plt.show()

    # Release memory.
    plt.clf()
    plt.close()
    
    #Result obtained after running the algo. Comment the below two lines if you want to run the algo
    mae_list.append(mae)
    comb.append(tag)  



train_X=diamonds_train[['x','cut','color']]# taking the training data features
train_y=diamonds_train[['lprice']] # output of our training data
test_X =diamonds_test[['x','cut','color']] # taking test data features
test_y=diamonds_test[['lprice']]  #output value of test data

# train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
# train_y=diamonds_train[['lprice']] # output of our training data
# test_X =diamonds_test[['carat','cut','clarity','color','x','y','z','depth','table']] # taking test data features
# test_y=diamonds_test[['lprice']]  #output value of test data



from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
    

#List of combinations
comb = []

#Dictionary to store the MAE for all algorithms 
mae_list = []

output_columns = ['Model','r^2 on test data', 'Model Score on Training Data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']
model_output = pd.DataFrame(columns=output_columns)

# print(model_output.head())


alpha = 1.0
linear = LinearRegression()
tag = 'linear'

pred_test = linear.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_linear = r2_score(test_y, pred_test)

# print(linear)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_linear)

model_r2_score = r2_score_linear

model_analysis(linear,tag,model_r2_score)


print('\n'*1)


#Ridge
alpha = 1.0
ridge = Ridge(alpha=alpha)
tag = 'ridge'

pred_test = ridge.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_ridge = r2_score(test_y, pred_test)

# print(ridge)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_ridge)

model_r2_score = r2_score_ridge

model_analysis(ridge,tag,model_r2_score)


print('\n'*1)


#Lasso

alpha = 0.1
lasso = Lasso(alpha=alpha)
tag = 'lasso'

pred_test = lasso.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_lasso = r2_score(test_y, pred_test)

# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

model_r2_score = r2_score_lasso

model_analysis(lasso,tag,model_r2_score)

print('\n'*1)

# alpha = 1.0
# lasso = Lasso(alpha=alpha)

# pred_test = lasso.fit(train_X, train_y).predict(test_X)
# r2_score_lasso = r2_score(test_y, pred_test)
# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

# print('\n'*1)

#ElasticNet

enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
tag = 'enet'

#y_pred_enet = enet.fit(train_X, train_y).predict(test_X)
pred_test= enet.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_enet = r2_score(test_y, pred_test)

# print(enet)
# print("r^2 on test data : %f" % r2_score_enet)

model_r2_score = r2_score_enet

model_analysis(enet,tag,model_r2_score)

#Unlike most other scores, R^2 score may be negative (it need not actually be the square of a quantity R).

print('\n'*1)
print ('Model formula : lprice ~ x + cut + color')

#print table
model_output.head()






# In[82]:

#http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_and_elasticnet.html#sphx-glr-auto-examples-linear-model-plot-lasso-and-elasticnet-py
#Working Lasso Example 

def model_analysis (model,tag,model_r2_score):


    mae = metrics.mean_absolute_error(test_y, predictions)
    mse = metrics.mean_squared_error(test_y, predictions)


    pred_len=int(len(predictions))

    output_columns = ['Model','r^2 on test data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']
 
    model_output.loc[model_output.shape[0]] = (tag,'{0:.6f}'.format(model_r2_score),'{0:.6f}'.format(model.score(test_X, test_y)),
                        '{0:.6f}'.format(model.score(train_X, train_y)),np.sqrt(metrics.mean_squared_error(test_y,predictions)),(mae),mse)
    
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    my_title=('Prediction for Model: %s'%(model))
    plt.title(my_title +'\n')
    #plt.title('Prediction')
    plt.scatter(test_y, pred_test, 
         label='Prediction Accuracy')
    plt.legend(loc='upper left')
    #plt.xlabel('Actual')
    #plt.ylabel('Predicted')

    plt.show()

    # Release memory.
    plt.clf()
    plt.close()
    
    #Result obtained after running the algo. Comment the below two lines if you want to run the algo
    mae_list.append(mae)
    comb.append(tag)  



train_X=diamonds_train[['y','cut','color']]# taking the training data features
train_y=diamonds_train[['lprice']] # output of our training data
test_X =diamonds_test[['y','cut','color']] # taking test data features
test_y=diamonds_test[['lprice']]  #output value of test data

# train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
# train_y=diamonds_train[['lprice']] # output of our training data
# test_X =diamonds_test[['carat','cut','clarity','color','x','y','z','depth','table']] # taking test data features
# test_y=diamonds_test[['lprice']]  #output value of test data



from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
    

#List of combinations
comb = []

#Dictionary to store the MAE for all algorithms 
mae_list = []

output_columns = ['Model','r^2 on test data', 'Model Score on Training Data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']
model_output = pd.DataFrame(columns=output_columns)

# print(model_output.head())


alpha = 1.0
linear = LinearRegression()
tag = 'linear'

pred_test = linear.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_linear = r2_score(test_y, pred_test)

# print(linear)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_linear)

model_r2_score = r2_score_linear

model_analysis(linear,tag,model_r2_score)


print('\n'*1)


#Ridge
alpha = 1.0
ridge = Ridge(alpha=alpha)
tag = 'ridge'

pred_test = ridge.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_ridge = r2_score(test_y, pred_test)

# print(ridge)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_ridge)

model_r2_score = r2_score_ridge

model_analysis(ridge,tag,model_r2_score)


print('\n'*1)


#Lasso

alpha = 0.1
lasso = Lasso(alpha=alpha)
tag = 'lasso alpha = 0.1'

pred_test = lasso.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_lasso = r2_score(test_y, pred_test)

# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

model_r2_score = r2_score_lasso

model_analysis(lasso,tag,model_r2_score)

print('\n'*1)

#Lasso 0.0001

alpha = 0.0001
lasso = Lasso(alpha=alpha)
tag = 'lasso alpha= 0.0001'

pred_test = lasso.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_lasso = r2_score(test_y, pred_test)

# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

model_r2_score = r2_score_lasso

model_analysis(lasso,tag,model_r2_score)

print('\n'*1)

#Lasso 0.0004

alpha = 0.00047
lasso = Lasso(alpha=alpha)
tag = 'lasso alpha= 0.00047'

pred_test = lasso.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_lasso = r2_score(test_y, pred_test)

# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

model_r2_score = r2_score_lasso

model_analysis(lasso,tag,model_r2_score)

print('\n'*1)

# alpha = 1.0
# lasso = Lasso(alpha=alpha)

# pred_test = lasso.fit(train_X, train_y).predict(test_X)
# r2_score_lasso = r2_score(test_y, pred_test)
# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

# print('\n'*1)

#ElasticNet
alpha = 0.1
enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
tag = 'enet alpha = 0.1'

#y_pred_enet = enet.fit(train_X, train_y).predict(test_X)
pred_test= enet.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_enet = r2_score(test_y, pred_test)

# print(enet)
# print("r^2 on test data : %f" % r2_score_enet)

model_r2_score = r2_score_enet

model_analysis(enet,tag,model_r2_score)

#ElasticNet
alpha = 0.0001
enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
tag = 'enet alpha = 0.0001'

#y_pred_enet = enet.fit(train_X, train_y).predict(test_X)
pred_test= enet.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_enet = r2_score(test_y, pred_test)

# print(enet)
# print("r^2 on test data : %f" % r2_score_enet)

model_r2_score = r2_score_enet

model_analysis(enet,tag,model_r2_score)

#ElasticNet
alpha = 0.00094
enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
tag = 'enet alpha = 0.00094'

#y_pred_enet = enet.fit(train_X, train_y).predict(test_X)
pred_test= enet.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_enet = r2_score(test_y, pred_test)

# print(enet)
# print("r^2 on test data : %f" % r2_score_enet)

model_r2_score = r2_score_enet

model_analysis(enet,tag,model_r2_score)


#Unlike most other scores, R^2 score may be negative (it need not actually be the square of a quantity R).

print('\n'*1)
print ('Model formula : lprice ~ y + cut + color')

#print table
model_output





# In[ ]:

Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)


# In[59]:

train

test


# In[110]:

# from sklearn.feature_selection import SelectFromModel
# from sklearn.svm import LinearSVC
# from sklearn.linear_model import LinearRegression
# from sklearn.pipeline import Pipeline

# X=np.array(train_X)
# y=np.array(train_y)

# #Reshape out Array from (37741L, 1L)) to 37741L,
# y = np.reshape(y, -1)




# ### Try out Regression with our R training dataset

# In[23]:

#Read in our Data
diamonds_train_R = pd.read_csv("diamonds_train_R.csv") #load the dataset
diamonds_test_R = pd.read_csv("diamonds_test_R.csv") #load the dataset


#Some Final Encoding based on #https://www.kaggle.com/enerrio/scikit-learn-ml-from-start-to-finish


from sklearn import preprocessing
def encode_features(df_train, df_test):
    features = ['cut','clarity','color']
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test
    
diamonds_train_R, diamonds_test_R = encode_features(diamonds_train_R, diamonds_test_R)

#diamonds_train.head()

# #Set up our training and test data
# train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
# train_y=diamonds_train[['lprice']] # output of our training data
# test_X =diamonds_test[['carat','cut','clarity','color','x','y','z','depth','table']] # taking test data features
# test_y=diamonds_test[['lprice']]  #output value of test data





# In[ ]:

#we know that the y,x,z  and carat are multicollinear
#so lests try the y only and then try carat as the sole predictor from out quartet


# #### Let's try y as out single value from out possible quartet of Multicollinear predictors

# In[47]:

print(np.log(325))


# In[48]:

print(np.exp(5.78))


# ### y-Model R split - show price plot

# In[58]:



def model_analysis (model,tag,model_r2_score):


    mae = metrics.mean_absolute_error(test_y, predictions)
    mse = metrics.mean_squared_error(test_y, predictions)


    pred_len=int(len(predictions))

    output_columns = ['Model','r^2 on test data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']
 
    model_output.loc[model_output.shape[0]] = (tag,'{0:.6f}'.format(model_r2_score),'{0:.6f}'.format(model.score(test_X, test_y)),
                        model.score(test_X, test_y),np.sqrt(metrics.mean_squared_error(test_y,predictions)),(mae),mse)
    
    
    #plt.figure(figsize=(12, 6))
    plt.figure(figsize=(6, 4))
    #plt.subplot(1, 2, 1)
    my_title=('Prediction for Model: %s'%(model))
    plt.title(my_title +'\n')
    #plt.title('Prediction')
    #plt.scatter(test_y, pred_test, 
         #label='Prediction Accuracy')
    plt.scatter(pred_test, test_y , 
         label='Prediction Accuracy')
    plt.legend(loc='upper left')
    #plt.xlabel('Actual')
    #plt.ylabel('Predicted')

    plt.show()
    
    plt.figure(figsize=(6, 4))
    #plt.subplot(1, 2, 2)
    my_title=('Price Prediction for Model: %s'%(model))
    plt.title(my_title +'\n')
    #plt.title('Prediction')
    #plt.scatter(np.exp(test_y), np.exp(pred_test), 
         #label='Prediction Accuracy')
    plt.scatter(np.exp(pred_test),np.exp(test_y), 
         label='Price Prediction')
    plt.legend(loc='lower right')
    
    plt.show()

    # Release memory.
    plt.clf()
    plt.close()
    
    #Result obtained after running the algo. Comment the below two lines if you want to run the algo
    mae_list.append(mae)
    comb.append(tag)  


# RFE gives us the following
# [(1.0, 'y'), 
#  (2.0, 'carat'), 
#  (3.0, 'z'), 
#  (4.0, 'x'), 
#  (5.0, 'clarity'), 
#  (6.0, 'color'), 
#  (7.0, 'depth'), 
#  (8.0, 'cut'), 
#  (9.0, 'table')]



train_X=diamonds_train_R[['y','clarity', 'color', 'depth',  'cut',  'table']]# taking the training data features
train_y=diamonds_train_R[['lprice']] # output of our training data
test_X =diamonds_test_R[['y','clarity', 'color', 'depth',  'cut',  'table']] # taking test data features
test_y=diamonds_test_R[['lprice']]  #output value of test data

# train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
# train_y=diamonds_train[['lprice']] # output of our training data
# test_X =diamonds_test[['carat','cut','clarity','color','x','y','z','depth','table']] # taking test data features
# test_y=diamonds_test[['lprice']]  #output value of test data



from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
    

#List of combinations
comb = []

#Dictionary to store the MAE for all algorithms 
mae_list = []

output_columns = ['Model','r^2 on test data', 'Model Score on Training Data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']
model_output = pd.DataFrame(columns=output_columns)

# print(model_output.head())


alpha = 1.0
linear = LinearRegression()
tag = 'linear'

pred_test = linear.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_linear = r2_score(test_y, pred_test)

# print(linear)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_linear)

model_r2_score = r2_score_linear

model_analysis(linear,tag,model_r2_score)

print('\n'*1)


#Ridge
alpha = 1.0
ridge = Ridge(alpha=alpha)
tag = 'ridge'

pred_test = ridge.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_ridge = r2_score(test_y, pred_test)

# print(ridge)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_ridge)

model_r2_score = r2_score_ridge

model_analysis(ridge,tag,model_r2_score)

print('\n'*1)


#Lasso

alpha = 0.1
lasso = Lasso(alpha=alpha)
tag = 'lasso'

pred_test = lasso.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_lasso = r2_score(test_y, pred_test)

# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

model_r2_score = r2_score_lasso

model_analysis(lasso,tag,model_r2_score)

print('\n'*1)

# alpha = 1.0
# lasso = Lasso(alpha=alpha)

# pred_test = lasso.fit(train_X, train_y).predict(test_X)
# r2_score_lasso = r2_score(test_y, pred_test)
# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

# print('\n'*1)

#ElasticNet

enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
tag = 'enet'

#y_pred_enet = enet.fit(train_X, train_y).predict(test_X)
pred_test= enet.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_enet = r2_score(test_y, pred_test)

# print(enet)
# print("r^2 on test data : %f" % r2_score_enet)

model_r2_score = r2_score_enet

model_analysis(enet,tag,model_r2_score)

#Unlike most other scores, R^2 score may be negative (it need not actually be the square of a quantity R).

print('\n'*1)

print ('Model formula : lprice ~ y + clarity + color + depth + cut + table')


model_output.head()





# ### Lets try all predictors on the R-split data

# In[24]:



def model_analysis (model,tag,model_r2_score):


    mae = metrics.mean_absolute_error(test_y, predictions)
    mse = metrics.mean_squared_error(test_y, predictions)


    pred_len=int(len(predictions))

    output_columns = ['Model','r^2 on test data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']
 
    model_output.loc[model_output.shape[0]] = (tag,'{0:.6f}'.format(model_r2_score),'{0:.6f}'.format(model.score(test_X, test_y)),
                        model.score(test_X, test_y),np.sqrt(metrics.mean_squared_error(test_y,predictions)),(mae),mse)
    
    
    #plt.figure(figsize=(12, 6))
    plt.figure(figsize=(6, 4))
    #plt.subplot(1, 2, 1)
    my_title=('Prediction for Model: %s'%(model))
    plt.title(my_title +'\n')
    #plt.title('Prediction')
    #plt.scatter(test_y, pred_test, 
         #label='Prediction Accuracy')
    plt.scatter(pred_test, test_y , 
         label='Prediction Accuracy')
    plt.legend(loc='upper left')
    #plt.xlabel('Actual')
    #plt.ylabel('Predicted')

    plt.show()
    
    plt.figure(figsize=(6, 4))
    #plt.subplot(1, 2, 2)
    my_title=('Price Prediction for Model: %s'%(model))
    plt.title(my_title +'\n')
    #plt.title('Prediction')
    #plt.scatter(np.exp(test_y), np.exp(pred_test), 
         #label='Prediction Accuracy')
    plt.scatter(np.exp(pred_test),np.exp(test_y), 
         label='Price Prediction')
    plt.legend(loc='lower right')
    
    plt.show()

    # Release memory.
    plt.clf()
    plt.close()
    
    #Result obtained after running the algo. Comment the below two lines if you want to run the algo
    mae_list.append(mae)
    comb.append(tag)  


# RFE gives us the following
# [(1.0, 'y'), 
#  (2.0, 'carat'), 
#  (3.0, 'z'), 
#  (4.0, 'x'), 
#  (5.0, 'clarity'), 
#  (6.0, 'color'), 
#  (7.0, 'depth'), 
#  (8.0, 'cut'), 
#  (9.0, 'table')]



train_X=diamonds_train_R[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
train_y=diamonds_train_R[['lprice']] # output of our training data
test_X =diamonds_test_R[['carat','cut','clarity','color','x','y','z','depth','table']] # taking test data features
test_y=diamonds_test_R[['lprice']]  #output value of test data

# train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
# train_y=diamonds_train[['lprice']] # output of our training data
# test_X =diamonds_test[['carat','cut','clarity','color','x','y','z','depth','table']] # taking test data features
# test_y=diamonds_test[['lprice']]  #output value of test data



from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
    

#List of combinations
comb = []

#Dictionary to store the MAE for all algorithms 
mae_list = []

output_columns = ['Model','r^2 on test data', 'Model Score on Training Data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']
model_output = pd.DataFrame(columns=output_columns)

# print(model_output.head())


alpha = 1.0
linear = LinearRegression()
tag = 'linear'

pred_test = linear.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_linear = r2_score(test_y, pred_test)

# print(linear)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_linear)

model_r2_score = r2_score_linear

model_analysis(linear,tag,model_r2_score)

print('\n'*1)


#Ridge
alpha = 1.0
ridge = Ridge(alpha=alpha)
tag = 'ridge'

pred_test = ridge.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_ridge = r2_score(test_y, pred_test)

# print(ridge)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_ridge)

model_r2_score = r2_score_ridge

model_analysis(ridge,tag,model_r2_score)

print('\n'*1)


#Lasso

alpha = 0.1
lasso = Lasso(alpha=alpha)
tag = 'lasso'

pred_test = lasso.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_lasso = r2_score(test_y, pred_test)

# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

model_r2_score = r2_score_lasso

model_analysis(lasso,tag,model_r2_score)

print('\n'*1)

# alpha = 1.0
# lasso = Lasso(alpha=alpha)

# pred_test = lasso.fit(train_X, train_y).predict(test_X)
# r2_score_lasso = r2_score(test_y, pred_test)
# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

# print('\n'*1)

#ElasticNet

enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
tag = 'enet'

#y_pred_enet = enet.fit(train_X, train_y).predict(test_X)
pred_test= enet.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_enet = r2_score(test_y, pred_test)

# print(enet)
# print("r^2 on test data : %f" % r2_score_enet)

model_r2_score = r2_score_enet

model_analysis(enet,tag,model_r2_score)

#Unlike most other scores, R^2 score may be negative (it need not actually be the square of a quantity R).

print('\n'*1)

print ('Model formula : lprice ~ y + clarity + color + depth + cut + table')


model_output.head()


# In[ ]:

###Lets try the same on the diamonds train/test data using all predictors for comparison with Rapidminer


# In[73]:



def model_analysis (model,tag,model_r2_score):


    mae = metrics.mean_absolute_error(test_y, predictions)
    mse = metrics.mean_squared_error(test_y, predictions)


    pred_len=int(len(predictions))

    output_columns = ['Model','r^2 on test data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']
 
    model_output.loc[model_output.shape[0]] = (tag,'{0:.6f}'.format(model_r2_score),'{0:.6f}'.format(model.score(test_X, test_y)),
                        model.score(test_X, test_y),np.sqrt(metrics.mean_squared_error(test_y,predictions)),(mae),mse)
    
    
    #plt.figure(figsize=(12, 6))
    plt.figure(figsize=(6, 4))
    #plt.subplot(1, 2, 1)
    my_title=('Prediction for Model: %s'%(model))
    plt.title(my_title +'\n')
    #plt.title('Prediction')
    #plt.scatter(test_y, pred_test, 
         #label='Prediction Accuracy')
    plt.scatter(pred_test, test_y , 
         label='Prediction Accuracy')
    plt.legend(loc='upper left')
    #plt.xlabel('Actual')
    #plt.ylabel('Predicted')

    plt.show()
    
    plt.figure(figsize=(6, 4))
    #plt.subplot(1, 2, 2)
    my_title=('Price Prediction for Model: %s'%(model))
    plt.title(my_title +'\n')
    #plt.title('Prediction')
    #plt.scatter(np.exp(test_y), np.exp(pred_test), 
         #label='Prediction Accuracy')
    plt.scatter(np.exp(pred_test),np.exp(test_y), 
         label='Price Prediction')
    plt.legend(loc='lower right')
    
    plt.show()

    # Release memory.
    plt.clf()
    plt.close()
    
    #Result obtained after running the algo. Comment the below two lines if you want to run the algo
    mae_list.append(mae)
    comb.append(tag)  


# RFE gives us the following
# [(1.0, 'y'), 
#  (2.0, 'carat'), 
#  (3.0, 'z'), 
#  (4.0, 'x'), 
#  (5.0, 'clarity'), 
#  (6.0, 'color'), 
#  (7.0, 'depth'), 
#  (8.0, 'cut'), 
#  (9.0, 'table')]



# train_X=diamonds_train_R[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
# train_y=diamonds_train_R[['lprice']] # output of our training data
# test_X =diamonds_test_R[['carat','cut','clarity','color','x','y','z','depth','table']] # taking test data features
# test_y=diamonds_test_R[['lprice']]  #output value of test data

train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
train_y=diamonds_train[['lprice']] # output of our training data
test_X =diamonds_test[['carat','cut','clarity','color','x','y','z','depth','table']] # taking test data features
test_y=diamonds_test[['lprice']]  #output value of test data



from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
    

#List of combinations
comb = []

#Dictionary to store the MAE for all algorithms 
mae_list = []

output_columns = ['Model','r^2 on test data', 'Model Score on Training Data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']
model_output = pd.DataFrame(columns=output_columns)

# print(model_output.head())


alpha = 1.0
linear = LinearRegression()
tag = 'linear'

pred_test = linear.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_linear = r2_score(test_y, pred_test)

# print(linear)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_linear)

model_r2_score = r2_score_linear

model_analysis(linear,tag,model_r2_score)

print('\n'*1)


#Ridge
alpha = 1.0
ridge = Ridge(alpha=alpha)
tag = 'ridge'

pred_test = ridge.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_ridge = r2_score(test_y, pred_test)

# print(ridge)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_ridge)

model_r2_score = r2_score_ridge

model_analysis(ridge,tag,model_r2_score)

print('\n'*1)


#Lasso

alpha = 0.1
lasso = Lasso(alpha=alpha)
tag = 'lasso'

pred_test = lasso.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_lasso = r2_score(test_y, pred_test)

# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

model_r2_score = r2_score_lasso

model_analysis(lasso,tag,model_r2_score)

print('\n'*1)

# alpha = 1.0
# lasso = Lasso(alpha=alpha)

# pred_test = lasso.fit(train_X, train_y).predict(test_X)
# r2_score_lasso = r2_score(test_y, pred_test)
# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

# print('\n'*1)

#ElasticNet

enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
tag = 'enet'

#y_pred_enet = enet.fit(train_X, train_y).predict(test_X)
pred_test= enet.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_enet = r2_score(test_y, pred_test)

# print(enet)
# print("r^2 on test data : %f" % r2_score_enet)

model_r2_score = r2_score_enet

model_analysis(enet,tag,model_r2_score)

#Unlike most other scores, R^2 score may be negative (it need not actually be the square of a quantity R).

print('\n'*1)

print ('Model formula : lprice ~ y + clarity + color + depth + cut + table')


model_output.head()


# In[ ]:

##Lprice Plots only


# In[26]:



def model_analysis (model,tag,model_r2_score):


    mae = metrics.mean_absolute_error(test_y, predictions)
    mse = metrics.mean_squared_error(test_y, predictions)


    pred_len=int(len(predictions))

    output_columns = ['Model','r^2 on test data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']
 
    model_output.loc[model_output.shape[0]] = (tag,'{0:.6f}'.format(model_r2_score),'{0:.6f}'.format(model.score(test_X, test_y)),
                        model.score(test_X, test_y),np.sqrt(metrics.mean_squared_error(test_y,predictions)),(mae),mse)
    
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    my_title=('Prediction for Model: %s'%(model))
    plt.title(my_title +'\n')
    #plt.title('Prediction')
    plt.scatter(test_y, pred_test, 
         label='Prediction Accuracy')
    plt.legend(loc='upper left')
    #plt.xlabel('Actual')
    #plt.ylabel('Predicted')

    plt.show()

    # Release memory.
    plt.clf()
    plt.close()
    
    #Result obtained after running the algo. Comment the below two lines if you want to run the algo
    mae_list.append(mae)
    comb.append(tag)  


# RFE gives us the following
# [(1.0, 'y'), 
#  (2.0, 'carat'), 
#  (3.0, 'z'), 
#  (4.0, 'x'), 
#  (5.0, 'clarity'), 
#  (6.0, 'color'), 
#  (7.0, 'depth'), 
#  (8.0, 'cut'), 
#  (9.0, 'table')]



train_X=diamonds_train_R[['y','clarity', 'color', 'depth',  'cut',  'table']]# taking the training data features
train_y=diamonds_train_R[['lprice']] # output of our training data
test_X =diamonds_test_R[['y','clarity', 'color', 'depth',  'cut',  'table']] # taking test data features
test_y=diamonds_test_R[['lprice']]  #output value of test data

# train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
# train_y=diamonds_train[['lprice']] # output of our training data
# test_X =diamonds_test[['carat','cut','clarity','color','x','y','z','depth','table']] # taking test data features
# test_y=diamonds_test[['lprice']]  #output value of test data



from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
    

#List of combinations
comb = []

#Dictionary to store the MAE for all algorithms 
mae_list = []

output_columns = ['Model','r^2 on test data', 'Model Score on Training Data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']
model_output = pd.DataFrame(columns=output_columns)

# print(model_output.head())


alpha = 1.0
linear = LinearRegression()
tag = 'linear'

pred_test = linear.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_linear = r2_score(test_y, pred_test)

# print(linear)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_linear)

model_r2_score = r2_score_linear

model_analysis(linear,tag,model_r2_score)

print('\n'*1)


#Ridge
alpha = 1.0
ridge = Ridge(alpha=alpha)
tag = 'ridge'

pred_test = ridge.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_ridge = r2_score(test_y, pred_test)

# print(ridge)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_ridge)

model_r2_score = r2_score_ridge

model_analysis(ridge,tag,model_r2_score)

print('\n'*1)


#Lasso

alpha = 0.1
lasso = Lasso(alpha=alpha)
tag = 'lasso'

pred_test = lasso.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_lasso = r2_score(test_y, pred_test)

# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

model_r2_score = r2_score_lasso

model_analysis(lasso,tag,model_r2_score)

print('\n'*1)

# alpha = 1.0
# lasso = Lasso(alpha=alpha)

# pred_test = lasso.fit(train_X, train_y).predict(test_X)
# r2_score_lasso = r2_score(test_y, pred_test)
# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

# print('\n'*1)

#ElasticNet

enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
tag = 'enet'

#y_pred_enet = enet.fit(train_X, train_y).predict(test_X)
pred_test= enet.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_enet = r2_score(test_y, pred_test)

# print(enet)
# print("r^2 on test data : %f" % r2_score_enet)

model_r2_score = r2_score_enet

model_analysis(enet,tag,model_r2_score)

#Unlike most other scores, R^2 score may be negative (it need not actually be the square of a quantity R).

print('\n'*1)

print ('Model formula : lprice ~ y + clarity + color + depth + cut + table')


model_output.head()






# In[ ]:

##### y-model R Split 


# In[85]:



def model_analysis (model,tag,model_r2_score,tag_code):


    mae = metrics.mean_absolute_error(test_y, predictions)
    mse = metrics.mean_squared_error(test_y, predictions)


    pred_len=int(len(predictions))

    output_columns = ['Model','r^2 on test data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']
 
    model_output.loc[model_output.shape[0]] = (tag,'{0:.6f}'.format(model_r2_score),'{0:.6f}'.format(model.score(test_X, test_y)),
                        '{0:.6f}'.format(model.score(train_X, train_y)),np.sqrt(metrics.mean_squared_error(test_y,predictions)),(mae),mse)
    
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    my_title=('Prediction for Model: %s'%(model))
    plt.title(my_title +'\n')
    #plt.title('Prediction')
    plt.scatter(test_y, pred_test, 
         label='Prediction Accuracy')
    plt.legend(loc='upper left')
    #plt.xlabel('Actual')
    #plt.ylabel('Predicted')

    plt.show()

    # Release memory.
    plt.clf()
    plt.close()
    
    #Result obtained after running the algo. Comment the below two lines if you want to run the algo
    mae_list.append(mae)
    comb.append(tag)  
    #predictions_comparison["'" + tag_code + "_" + "actual" + "'"] = test_y
    #predictions_comparison["'" + tag_code + "_" + "predicted" + "'"] = predictions
    predictions_comparison[tag_code + "_a"] = test_y
    predictions_comparison[tag_code + "_p" ] = predictions
    #tag.replace(" ", "_")

# RFE gives us the following
# [(1.0, 'y'), 
#  (2.0, 'carat'), 
#  (3.0, 'z'), 
#  (4.0, 'x'), 
#  (5.0, 'clarity'), 
#  (6.0, 'color'), 
#  (7.0, 'depth'), 
#  (8.0, 'cut'), 
#  (9.0, 'table')]



train_X=diamonds_train_R[['y','clarity', 'color', 'depth',  'cut',  'table']]# taking the training data features
train_y=diamonds_train_R[['lprice']] # output of our training data
test_X =diamonds_test_R[['y','clarity', 'color', 'depth',  'cut',  'table']] # taking test data features
test_y=diamonds_test_R[['lprice']]  #output value of test data

# train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
# train_y=diamonds_train[['lprice']] # output of our training data
# test_X =diamonds_test[['carat','cut','clarity','color','x','y','z','depth','table']] # taking test data features
# test_y=diamonds_test[['lprice']]  #output value of test data



from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
    

#List of combinations
comb = []

#Dictionary to store the MAE for all algorithms 
mae_list = []

output_columns = ['Model','r^2 on test data', 'Model Score on Training Data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']
model_output = pd.DataFrame(columns=output_columns)

predictions_comparison =pd.DataFrame()

# print(model_output.head())


alpha = 1.0
linear = LinearRegression()
tag = 'linear'
tag_code = 'linear'
pred_test = linear.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_linear = r2_score(test_y, pred_test)

# print(linear)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_linear)

model_r2_score = r2_score_linear

model_analysis(linear,tag,model_r2_score,tag_code)

print('\n'*1)


#Ridge
alpha = 1.0
ridge = Ridge(alpha=alpha)
tag = 'ridge'
tag_code = 'ridge'
pred_test = ridge.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_ridge = r2_score(test_y, pred_test)

# print(ridge)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_ridge)

model_r2_score = r2_score_ridge

model_analysis(ridge,tag,model_r2_score,tag_code)

print('\n'*1)


#Lasso

alpha = 0.1
lasso = Lasso(alpha=alpha)
tag = 'lasso alpha = 0.1'
tag_code = 'lasso_0_1'
pred_test = lasso.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_lasso = r2_score(test_y, pred_test)

# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

model_r2_score = r2_score_lasso

model_analysis(lasso,tag,model_r2_score,tag_code)

print('\n'*1)

#Lasso 0.0001

alpha = 0.0001
lasso = Lasso(alpha=alpha)
tag = 'lasso alpha = 0.0001'
tag_code ='lasso_0_0001'

pred_test = lasso.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_lasso = r2_score(test_y, pred_test)

# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

model_r2_score = r2_score_lasso

model_analysis(lasso,tag,model_r2_score,tag_code)

print('\n'*1)

#Lasso 0.0004

alpha = 0.00047
lasso = Lasso(alpha=alpha)
tag = 'lasso alpha = 0.00047'
tag_code = 'lasso_0_00047'

pred_test = lasso.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_lasso = r2_score(test_y, pred_test)

# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

model_r2_score = r2_score_lasso

model_analysis(lasso,tag,model_r2_score,tag_code)

print('\n'*1)

# alpha = 1.0
# lasso = Lasso(alpha=alpha)

# pred_test = lasso.fit(train_X, train_y).predict(test_X)
# r2_score_lasso = r2_score(test_y, pred_test)
# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

# print('\n'*1)

#ElasticNet
alpha = 0.1
enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
tag = 'enet alpha = 0.1'
tag_code = 'enet_0_1'

#y_pred_enet = enet.fit(train_X, train_y).predict(test_X)
pred_test= enet.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_enet = r2_score(test_y, pred_test)

# print(enet)
# print("r^2 on test data : %f" % r2_score_enet)

model_r2_score = r2_score_enet

model_analysis(enet,tag,model_r2_score,tag_code)


#ElasticNet
alpha = 0.0001
enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
tag = 'enet alpha = 0.0001'
tag_code = 'enet_0_0001'

#y_pred_enet = enet.fit(train_X, train_y).predict(test_X)
pred_test= enet.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_enet = r2_score(test_y, pred_test)

# print(enet)
# print("r^2 on test data : %f" % r2_score_enet)

model_r2_score = r2_score_enet

model_analysis(enet,tag,model_r2_score,tag_code)

#ElasticNet
alpha = 0.00094
enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
tag = 'enet alpha = 0.00094'
tag_code = 'enet_0_00094'

#y_pred_enet = enet.fit(train_X, train_y).predict(test_X)
pred_test= enet.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_enet = r2_score(test_y, pred_test)

# print(enet)
# print("r^2 on test data : %f" % r2_score_enet)

model_r2_score = r2_score_enet

model_analysis(enet,tag,model_r2_score,tag_code)

#Unlike most other scores, R^2 score may be negative (it need not actually be the square of a quantity R).

print('\n'*1)

print ('Model formula : lprice ~ y + clarity + color + depth + cut + table')


model_output


# In[ ]:

#### y-Models R split  



# In[36]:

repl_test ='lasso alpha = 0.1_actual'
repl_test.replace(' ','')
repl_test.replace(' ','_')
print(repl_test)


# In[86]:

predictions_comparison.head(5)


# In[87]:

# Lets look at the distributions

#https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
#plt.yaxis=dict(range=[0, 7])

plt.figure(figsize=(12,10))
plt.subplot(3,3,1)
sns.distplot(predictions_comparison.linear_p)
plt.subplot(3,3,2)
sns.distplot(predictions_comparison.ridge_p)
plt.subplot(3,3,3)
sns.distplot(predictions_comparison.lasso_0_1_p)
plt.subplot(3,3,4)
sns.distplot(predictions_comparison.lasso_0_0001_p)
plt.subplot(3,3,5)
sns.distplot(predictions_comparison.lasso_0_00047_p)
plt.subplot(3,3,6)
sns.distplot(predictions_comparison.enet_0_1_p)
plt.subplot(3,3,7)
sns.distplot(predictions_comparison.enet_0_0001_p)
plt.subplot(3,3,8)
sns.distplot(predictions_comparison.enet_0_00094_p)
plt.subplot(3,3,9)
sns.distplot(predictions_comparison.linear_a)
plt.show()
# Release memory.
plt.clf()
plt.close()


# In[88]:

import scipy

def shaipro_test_normailty(data,tag_code):

    shapiro_results = scipy.stats.shapiro(data)

#     matrix_sw = [
#         ['', 'DF', 'Test Statistic', 'p-value'],
#         ['Sample Data', len(x) - 1, shapiro_results[0], shapiro_results[1]]
#     ]
        
    shapiro_output.loc[shapiro_output.shape[0]] = (tag_code,(len(data) - 1),shapiro_results[0],shapiro_results[1])
    
    #shapiro_output    
    #shapiro_table = FF.create_table(matrix_sw, index=True)
    #py.iplot(shapiro_table, filename='shapiro-table')

    
shapiro_columns = ['Model', 'DF', 'Test Statistic', 'p-value']
shapiro_output = pd.DataFrame(columns=shapiro_columns)
    
    
    #['Model', 'DF', 'Test Statistic', 'p-value']
shaipro_test_normailty(predictions_comparison.linear_p,'linear_p')
shaipro_test_normailty(predictions_comparison.ridge_p,'ridge_p')
shaipro_test_normailty(predictions_comparison.lasso_0_1_p,'lasso_0_1_p')
shaipro_test_normailty(predictions_comparison.lasso_0_0001_p,'lasso_0_0001_p')
shaipro_test_normailty(predictions_comparison.lasso_0_00047_p,'lasso_0_00047_p')
shaipro_test_normailty(predictions_comparison.enet_0_1_p,'enet_0_1_p')
shaipro_test_normailty(predictions_comparison.enet_0_0001_p,'enet_0_0001_p')
shaipro_test_normailty(predictions_comparison.enet_0_00094_p,'enet_0_00094_p')
shaipro_test_normailty(predictions_comparison.linear_a,'linear_a')

shapiro_output




#Shapiro


# In[ ]:

##Kolmogorov-Smirnov NULL HUPOTHESIS data is Normally Distributed


# In[93]:

import scipy

def kolmogorov_smirnov_test_normailty(data,tag_code):

    kolmogorov_smirnov_results = scipy.stats.kstest(data, cdf='norm')

#     matrix_sw = [
#         ['', 'DF', 'Test Statistic', 'p-value'],
#         ['Sample Data', len(x) - 1, shapiro_results[0], shapiro_results[1]]
#     ]
        
    kolmogorov_smirnov_output.loc[kolmogorov_smirnov_output.shape[0]] = (tag_code,(len(data) - 1),kolmogorov_smirnov_results[0],kolmogorov_smirnov_results[1])
    
    
kolmogorov_smirnov_columns = ['Model', 'DF', 'Test Statistic', 'p-value']
kolmogorov_smirnov_output = pd.DataFrame(columns=kolmogorov_smirnov_columns)
    
    
#['Model', 'DF', 'Test Statistic', 'p-value']
kolmogorov_smirnov_test_normailty(predictions_comparison.linear_p,'linear_p')
kolmogorov_smirnov_test_normailty(predictions_comparison.ridge_p,'ridge_p')
kolmogorov_smirnov_test_normailty(predictions_comparison.lasso_0_1_p,'lasso_0_1_p')
kolmogorov_smirnov_test_normailty(predictions_comparison.lasso_0_0001_p,'lasso_0_0001_p')
kolmogorov_smirnov_test_normailty(predictions_comparison.lasso_0_00047_p,'lasso_0_00047_p')
kolmogorov_smirnov_test_normailty(predictions_comparison.enet_0_1_p,'enet_0_1_p')
kolmogorov_smirnov_test_normailty(predictions_comparison.enet_0_0001_p,'enet_0_0001_p')
kolmogorov_smirnov_test_normailty(predictions_comparison.enet_0_00094_p,'enet_0_00094_p')
kolmogorov_smirnov_test_normailty(predictions_comparison.linear_a,'linear_a')

kolmogorov_smirnov_output


#Kolmogorov-Smirnov


# In[ ]:

##Since our p-value is read as 0.0 (meaning it is "practically" 0 given the decimal accuracy of the test) 
#then we have strong evidence to not reject the null-hypothesis - No evidence that it is Not Normally Distributed


# In[89]:

## Now test if distributions are different

# test

from scipy import stats


def t_test_compare(data,tag_code,flag_test) :
    if flag_test == 0:
        compare_test_results = stats.ttest_ind(data,predictions_comparison.linear_a)
        test_name = 'Student’s t-test'
    else:
        compare_test_results = stats.ttest_ind(data,predictions_comparison.linear_a, equal_var = False)
        test_name = 'Welch’s t-test'
    t_test_output.loc[t_test_output.shape[0]] = (tag_code,compare_test_results[0],compare_test_results[1],test_name)
    
    
t_test_columns = ['Model', 'Test Statistic', 'p-value', ' Test']
t_test_output = pd.DataFrame(columns=t_test_columns)
    
    
#['Model', 'DF', 'Test Statistic', 'p-value']
t_test_compare(predictions_comparison.linear_p,'linear_p',0)
t_test_compare(predictions_comparison.linear_p,'linear_p',1)

t_test_compare(predictions_comparison.ridge_p,'ridge_p',0)
t_test_compare(predictions_comparison.ridge_p,'ridge_p',1)

t_test_compare(predictions_comparison.lasso_0_1_p,'lasso_0_1_p',0)
t_test_compare(predictions_comparison.lasso_0_1_p,'lasso_0_1_p',1)

t_test_compare(predictions_comparison.lasso_0_0001_p,'lasso_0_0001_p',0)
t_test_compare(predictions_comparison.lasso_0_0001_p,'lasso_0_0001_p',1)

t_test_compare(predictions_comparison.lasso_0_00047_p,'lasso_0_00047_p',0)
t_test_compare(predictions_comparison.lasso_0_00047_p,'lasso_0_00047_p',1)

t_test_compare(predictions_comparison.enet_0_1_p,'enet_0_1_p',0)
t_test_compare(predictions_comparison.enet_0_1_p,'enet_0_1_p',1)

t_test_compare(predictions_comparison.enet_0_0001_p,'enet_0_0001_p',0)
t_test_compare(predictions_comparison.enet_0_0001_p,'enet_0_0001_p',1)

t_test_compare(predictions_comparison.enet_0_00094_p,'enet_0_00094_p',0)
t_test_compare(predictions_comparison.enet_0_00094_p,'enet_0_00094_p',1)

#t_test_compare(predictions_comparison.linear_a,'linear_a')

t_test_output

#Test with sample with identical means:

#If we observe a large p-value, for example larger than 0.05 or 0.1, 
#then we cannot reject the null hypothesis of identical average scores. 
#If the p-value is smaller than the threshold, e.g. 1%, 5% or 10%, then we reject the null hypothesis of equal averages.


# In[82]:

tag


# In[91]:

# Lets look at the distributions

#https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python


plt.figure(figsize=(24,20))
plt.subplot(3,3,1)
sns.distplot(predictions_comparison.linear_p)
sns.distplot(predictions_comparison.ridge_p)
sns.distplot(predictions_comparison.lasso_0_1_p)
sns.distplot(predictions_comparison.enet_0_1_p)
plt.show()
# Release memory.
plt.clf()
plt.close()


# In[ ]:

######## Marker ~~~~~ #####


# In[ ]:




# #### Let's try carat as out single value from out possible quatet of Multicollinear predictors

# In[27]:



def model_analysis (model,tag,model_r2_score):


    mae = metrics.mean_absolute_error(test_y, predictions)
    mse = metrics.mean_squared_error(test_y, predictions)


    pred_len=int(len(predictions))

    output_columns = ['Model','r^2 on test data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']
 
    model_output.loc[model_output.shape[0]] = (tag,'{0:.6f}'.format(model_r2_score),'{0:.6f}'.format(model.score(test_X, test_y)),
                        model.score(test_X, test_y),np.sqrt(metrics.mean_squared_error(test_y,predictions)),(mae),mse)
    
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    my_title=('Prediction for Model: %s'%(model))
    plt.title(my_title +'\n')
    #plt.title('Prediction')
    plt.scatter(test_y, pred_test, 
         label='Prediction Accuracy')
    plt.legend(loc='upper left')
    #plt.xlabel('Actual')
    #plt.ylabel('Predicted')

    plt.show()

    # Release memory.
    plt.clf()
    plt.close()
    
    #Result obtained after running the algo. Comment the below two lines if you want to run the algo
    mae_list.append(mae)
    comb.append(tag)  


# RFE gives us the following
# [(1.0, 'y'), 
#  (2.0, 'carat'), 
#  (3.0, 'z'), 
#  (4.0, 'x'), 
#  (5.0, 'clarity'), 
#  (6.0, 'color'), 
#  (7.0, 'depth'), 
#  (8.0, 'cut'), 
#  (9.0, 'table')]



train_X=diamonds_train_R[['carat','clarity', 'color', 'depth',  'cut',  'table']]# taking the training data features
train_y=diamonds_train_R[['lprice']] # output of our training data
test_X =diamonds_test_R[['carat','clarity', 'color', 'depth',  'cut',  'table']] # taking test data features
test_y=diamonds_test_R[['lprice']]  #output value of test data

# train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
# train_y=diamonds_train[['lprice']] # output of our training data
# test_X =diamonds_test[['carat','cut','clarity','color','x','y','z','depth','table']] # taking test data features
# test_y=diamonds_test[['lprice']]  #output value of test data



from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
    

#List of combinations
comb = []

#Dictionary to store the MAE for all algorithms 
mae_list = []

output_columns = ['Model','r^2 on test data', 'Model Score on Training Data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']
model_output = pd.DataFrame(columns=output_columns)

# print(model_output.head())


alpha = 1.0
linear = LinearRegression()
tag = 'linear'

pred_test = linear.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_linear = r2_score(test_y, pred_test)

# print(linear)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_linear)

model_r2_score = r2_score_linear

model_analysis(linear,tag,model_r2_score)

print('\n'*1)


#Ridge
alpha = 1.0
ridge = Ridge(alpha=alpha)
tag = 'ridge'

pred_test = ridge.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_ridge = r2_score(test_y, pred_test)

# print(ridge)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_ridge)

model_r2_score = r2_score_ridge

model_analysis(ridge,tag,model_r2_score)

print('\n'*1)


#Lasso

alpha = 0.1
lasso = Lasso(alpha=alpha)
tag = 'lasso'

pred_test = lasso.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_lasso = r2_score(test_y, pred_test)

# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

model_r2_score = r2_score_lasso

model_analysis(lasso,tag,model_r2_score)

print('\n'*1)

# alpha = 1.0
# lasso = Lasso(alpha=alpha)

# pred_test = lasso.fit(train_X, train_y).predict(test_X)
# r2_score_lasso = r2_score(test_y, pred_test)
# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

# print('\n'*1)

#ElasticNet

enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
tag = 'enet'

#y_pred_enet = enet.fit(train_X, train_y).predict(test_X)
pred_test= enet.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_enet = r2_score(test_y, pred_test)

# print(enet)
# print("r^2 on test data : %f" % r2_score_enet)

model_r2_score = r2_score_enet

model_analysis(enet,tag,model_r2_score)

#Unlike most other scores, R^2 score may be negative (it need not actually be the square of a quantity R).

print('\n'*1)

print ('Model formula : lprice ~ carat + clarity + color + depth + cut + table')


model_output.head()



# In[92]:



def model_analysis (model,tag,model_r2_score):


    mae = metrics.mean_absolute_error(test_y, predictions)
    mse = metrics.mean_squared_error(test_y, predictions)


    pred_len=int(len(predictions))

    output_columns = ['Model','r^2 on test data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']
 
    model_output.loc[model_output.shape[0]] = (tag,'{0:.6f}'.format(model_r2_score),'{0:.6f}'.format(model.score(test_X, test_y)),
                        '{0:.6f}'.format(model.score(train_X, train_y)),np.sqrt(metrics.mean_squared_error(test_y,predictions)),(mae),mse)
    
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    my_title=('Prediction for Model: %s'%(model))
    plt.title(my_title +'\n')
    #plt.title('Prediction')
    plt.scatter(test_y, pred_test, 
         label='Prediction Accuracy')
    plt.legend(loc='upper left')
    #plt.xlabel('Actual')
    #plt.ylabel('Predicted')

    plt.show()

    # Release memory.
    plt.clf()
    plt.close()
    
    #Result obtained after running the algo. Comment the below two lines if you want to run the algo
    mae_list.append(mae)
    comb.append(tag)  


# RFE gives us the following
# [(1.0, 'y'), 
#  (2.0, 'carat'), 
#  (3.0, 'z'), 
#  (4.0, 'x'), 
#  (5.0, 'clarity'), 
#  (6.0, 'color'), 
#  (7.0, 'depth'), 
#  (8.0, 'cut'), 
#  (9.0, 'table')]



train_X=diamonds_train_R[['carat','clarity', 'color', 'depth',  'cut',  'table']]# taking the training data features
train_y=diamonds_train_R[['lprice']] # output of our training data
test_X =diamonds_test_R[['carat','clarity', 'color', 'depth',  'cut',  'table']] # taking test data features
test_y=diamonds_test_R[['lprice']]  #output value of test data

# train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
# train_y=diamonds_train[['lprice']] # output of our training data
# test_X =diamonds_test[['carat','cut','clarity','color','x','y','z','depth','table']] # taking test data features
# test_y=diamonds_test[['lprice']]  #output value of test data



from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
    

#List of combinations
comb = []

#Dictionary to store the MAE for all algorithms 
mae_list = []

output_columns = ['Model','r^2 on test data', 'Model Score on Training Data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']
model_output = pd.DataFrame(columns=output_columns)

# print(model_output.head())


alpha = 1.0
linear = LinearRegression()
tag = 'linear'

pred_test = linear.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_linear = r2_score(test_y, pred_test)

# print(linear)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_linear)

model_r2_score = r2_score_linear

model_analysis(linear,tag,model_r2_score)

print('\n'*1)


#Ridge
alpha = 1.0
ridge = Ridge(alpha=alpha)
tag = 'ridge'

pred_test = ridge.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_ridge = r2_score(test_y, pred_test)

# print(ridge)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_ridge)

model_r2_score = r2_score_ridge

model_analysis(ridge,tag,model_r2_score)

print('\n'*1)


#Lasso

alpha = 0.1
lasso = Lasso(alpha=alpha)
tag = 'lasso alpha = 0.1'

pred_test = lasso.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_lasso = r2_score(test_y, pred_test)

# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

model_r2_score = r2_score_lasso

model_analysis(lasso,tag,model_r2_score)

print('\n'*1)

#Lasso 0.0001

alpha = 0.0001
lasso = Lasso(alpha=alpha)
tag = 'lasso alpha= 0.0001'

pred_test = lasso.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_lasso = r2_score(test_y, pred_test)

# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

model_r2_score = r2_score_lasso

model_analysis(lasso,tag,model_r2_score)

print('\n'*1)

#Lasso 0.0004

alpha = 0.00047
lasso = Lasso(alpha=alpha)
tag = 'lasso alpha= 0.00047'

pred_test = lasso.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_lasso = r2_score(test_y, pred_test)

# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

model_r2_score = r2_score_lasso

model_analysis(lasso,tag,model_r2_score)

print('\n'*1)

# alpha = 1.0
# lasso = Lasso(alpha=alpha)

# pred_test = lasso.fit(train_X, train_y).predict(test_X)
# r2_score_lasso = r2_score(test_y, pred_test)
# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

# print('\n'*1)

#ElasticNet
alpha = 0.1
enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
tag = 'enet alpha = 0.1'

#y_pred_enet = enet.fit(train_X, train_y).predict(test_X)
pred_test= enet.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_enet = r2_score(test_y, pred_test)

# print(enet)
# print("r^2 on test data : %f" % r2_score_enet)

model_r2_score = r2_score_enet

model_analysis(enet,tag,model_r2_score)


#ElasticNet
alpha = 0.0001
enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
tag = 'enet alpha = 0.0001'

#y_pred_enet = enet.fit(train_X, train_y).predict(test_X)
pred_test= enet.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_enet = r2_score(test_y, pred_test)

# print(enet)
# print("r^2 on test data : %f" % r2_score_enet)

model_r2_score = r2_score_enet

model_analysis(enet,tag,model_r2_score)

#ElasticNet
alpha = 0.00094
enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
tag = 'enet alpha = 0.00094'

#y_pred_enet = enet.fit(train_X, train_y).predict(test_X)
pred_test= enet.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_enet = r2_score(test_y, pred_test)

# print(enet)
# print("r^2 on test data : %f" % r2_score_enet)

model_r2_score = r2_score_enet

model_analysis(enet,tag,model_r2_score)


#Unlike most other scores, R^2 score may be negative (it need not actually be the square of a quantity R).

print('\n'*1)

print ('Model formula : lprice ~ carat + clarity + color + depth + cut + table')


model_output


# In[ ]:

##### Let try y and carat - and see if the effects of multicollinearity are visible


# In[35]:



def model_analysis (model,tag,model_r2_score):


    mae = metrics.mean_absolute_error(test_y, predictions)
    mse = metrics.mean_squared_error(test_y, predictions)


    pred_len=int(len(predictions))

    output_columns = ['Model','r^2 on test data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']
 
    model_output.loc[model_output.shape[0]] = (tag,'{0:.6f}'.format(model_r2_score),'{0:.6f}'.format(model.score(test_X, test_y)),
                        model.score(test_X, test_y),np.sqrt(metrics.mean_squared_error(test_y,predictions)),(mae),mse)
    
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    my_title=('Prediction for Model: %s'%(model))
    plt.title(my_title +'\n')
    #plt.title('Prediction')
    plt.scatter(test_y, pred_test, 
         label='Prediction Accuracy')
    plt.legend(loc='upper left')
    #plt.xlabel('Actual')
    #plt.ylabel('Predicted')

    plt.show()

    # Release memory.
    plt.clf()
    plt.close()
    
    #Result obtained after running the algo. Comment the below two lines if you want to run the algo
    mae_list.append(mae)
    comb.append(tag)  


# RFE gives us the following
# [(1.0, 'y'), 
#  (2.0, 'carat'), 
#  (3.0, 'z'), 
#  (4.0, 'x'), 
#  (5.0, 'clarity'), 
#  (6.0, 'color'), 
#  (7.0, 'depth'), 
#  (8.0, 'cut'), 
#  (9.0, 'table')]



train_X=diamonds_train_R[['y','carat','clarity', 'color', 'depth',  'cut',  'table']]# taking the training data features
train_y=diamonds_train_R[['lprice']] # output of our training data
test_X =diamonds_test_R[['y','carat','clarity', 'color', 'depth',  'cut',  'table']] # taking test data features
test_y=diamonds_test_R[['lprice']]  #output value of test data

# train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
# train_y=diamonds_train[['lprice']] # output of our training data
# test_X =diamonds_test[['carat','cut','clarity','color','x','y','z','depth','table']] # taking test data features
# test_y=diamonds_test[['lprice']]  #output value of test data



from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
    

#List of combinations
comb = []

#Dictionary to store the MAE for all algorithms 
mae_list = []

output_columns = ['Model','r^2 on test data', 'Model Score on Training Data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']
model_output = pd.DataFrame(columns=output_columns)

# print(model_output.head())


alpha = 1.0
linear = LinearRegression()
tag = 'linear'

pred_test = linear.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_linear = r2_score(test_y, pred_test)

# print(linear)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_linear)

model_r2_score = r2_score_linear

model_analysis(linear,tag,model_r2_score)

print('\n'*1)


#Ridge
alpha = 1.0
ridge = Ridge(alpha=alpha)
tag = 'ridge'

pred_test = ridge.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_ridge = r2_score(test_y, pred_test)

# print(ridge)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_ridge)

model_r2_score = r2_score_ridge

model_analysis(ridge,tag,model_r2_score)

print('\n'*1)


#Lasso

alpha = 0.1
lasso = Lasso(alpha=alpha)
tag = 'lasso'

pred_test = lasso.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_lasso = r2_score(test_y, pred_test)

# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

model_r2_score = r2_score_lasso

model_analysis(lasso,tag,model_r2_score)

print('\n'*1)

# alpha = 1.0
# lasso = Lasso(alpha=alpha)

# pred_test = lasso.fit(train_X, train_y).predict(test_X)
# r2_score_lasso = r2_score(test_y, pred_test)
# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

# print('\n'*1)

#ElasticNet

enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
tag = 'enet'

#y_pred_enet = enet.fit(train_X, train_y).predict(test_X)
pred_test= enet.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_enet = r2_score(test_y, pred_test)

# print(enet)
# print("r^2 on test data : %f" % r2_score_enet)

model_r2_score = r2_score_enet

model_analysis(enet,tag,model_r2_score)

#Unlike most other scores, R^2 score may be negative (it need not actually be the square of a quantity R).

print('\n'*1)

print ('Model formula : lprice ~ y + carat + clarity + color + depth + cut + table')


model_output.head()


# In[ ]:

###### y - Models R Split + carat


# In[93]:

def model_analysis (model,tag,model_r2_score):


    mae = metrics.mean_absolute_error(test_y, predictions)
    mse = metrics.mean_squared_error(test_y, predictions)


    pred_len=int(len(predictions))

    output_columns = ['Model','r^2 on test data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']
 
    model_output.loc[model_output.shape[0]] = (tag,'{0:.6f}'.format(model_r2_score),'{0:.6f}'.format(model.score(test_X, test_y)),
                        '{0:.6f}'.format(model.score(train_X, train_y)),np.sqrt(metrics.mean_squared_error(test_y,predictions)),(mae),mse)
    
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    my_title=('Prediction for Model: %s'%(model))
    plt.title(my_title +'\n')
    #plt.title('Prediction')
    plt.scatter(test_y, pred_test, 
         label='Prediction Accuracy')
    plt.legend(loc='upper left')
    #plt.xlabel('Actual')
    #plt.ylabel('Predicted')

    plt.show()

    # Release memory.
    plt.clf()
    plt.close()
    
    #Result obtained after running the algo. Comment the below two lines if you want to run the algo
    mae_list.append(mae)
    comb.append(tag)  

    predictions_comparison["'" + tag + "_" + "actual" + "'"] = test_y
    predictions_comparison["'" + tag + "_" + "predicted" + "'"] = predictions

# RFE gives us the following
# [(1.0, 'y'), 
#  (2.0, 'carat'), 
#  (3.0, 'z'), 
#  (4.0, 'x'), 
#  (5.0, 'clarity'), 
#  (6.0, 'color'), 
#  (7.0, 'depth'), 
#  (8.0, 'cut'), 
#  (9.0, 'table')]



train_X=diamonds_train_R[['y','carat','clarity', 'color', 'depth',  'cut',  'table']]# taking the training data features
train_y=diamonds_train_R[['lprice']] # output of our training data
test_X =diamonds_test_R[['y','carat','clarity', 'color', 'depth',  'cut',  'table']] # taking test data features
test_y=diamonds_test_R[['lprice']]  #output value of test data

# train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
# train_y=diamonds_train[['lprice']] # output of our training data
# test_X =diamonds_test[['carat','cut','clarity','color','x','y','z','depth','table']] # taking test data features
# test_y=diamonds_test[['lprice']]  #output value of test data



from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
    

#List of combinations
comb = []

#Dictionary to store the MAE for all algorithms 
mae_list = []


predictions_list = []

output_columns = ['Model','r^2 on test data', 'Model Score on Training Data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']
model_output = pd.DataFrame(columns=output_columns)

predictions_comparison =pd.DataFrame()



# print(model_output.head())


alpha = 1.0
linear = LinearRegression()
tag = 'linear'

pred_test = linear.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_linear = r2_score(test_y, pred_test)

# print(linear)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_linear)

model_r2_score = r2_score_linear

model_analysis(linear,tag,model_r2_score)

print('\n'*1)


#Ridge
alpha = 1.0
ridge = Ridge(alpha=alpha)
tag = 'ridge'

pred_test = ridge.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_ridge = r2_score(test_y, pred_test)

# print(ridge)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_ridge)

model_r2_score = r2_score_ridge

model_analysis(ridge,tag,model_r2_score)

print('\n'*1)


#Lasso

alpha = 0.1
lasso = Lasso(alpha=alpha)
tag = 'lasso alpha = 0.1'

pred_test = lasso.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_lasso = r2_score(test_y, pred_test)

# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

model_r2_score = r2_score_lasso

model_analysis(lasso,tag,model_r2_score)

print('\n'*1)


#Lasso 0.0001

alpha = 0.0001
lasso = Lasso(alpha=alpha)
tag = 'lasso alpha = 0.0001'

pred_test = lasso.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_lasso = r2_score(test_y, pred_test)

# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

model_r2_score = r2_score_lasso

model_analysis(lasso,tag,model_r2_score)

print('\n'*1)

#Lasso 0.0004

alpha = 0.00047
lasso = Lasso(alpha=alpha)
tag = 'lasso alpha = 0.00047'

pred_test = lasso.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_lasso = r2_score(test_y, pred_test)

# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

model_r2_score = r2_score_lasso

model_analysis(lasso,tag,model_r2_score)

print('\n'*1)

# alpha = 1.0
# lasso = Lasso(alpha=alpha)

# pred_test = lasso.fit(train_X, train_y).predict(test_X)
# r2_score_lasso = r2_score(test_y, pred_test)
# print(lasso)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_lasso)

# print('\n'*1)

#ElasticNet
alpha = 0.1
enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
tag = 'enet alpha = 0.1'

#y_pred_enet = enet.fit(train_X, train_y).predict(test_X)
pred_test= enet.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_enet = r2_score(test_y, pred_test)

# print(enet)
# print("r^2 on test data : %f" % r2_score_enet)

model_r2_score = r2_score_enet

model_analysis(enet,tag,model_r2_score)


#ElasticNet
alpha = 0.0001
enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
tag = 'enet alpha = 0.0001'

#y_pred_enet = enet.fit(train_X, train_y).predict(test_X)
pred_test= enet.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_enet = r2_score(test_y, pred_test)

# print(enet)
# print("r^2 on test data : %f" % r2_score_enet)

model_r2_score = r2_score_enet

model_analysis(enet,tag,model_r2_score)

#0.00094373230442450738

#ElasticNet
alpha = 0.00094
enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
tag = 'enet alpha = 0.00094'

#y_pred_enet = enet.fit(train_X, train_y).predict(test_X)
pred_test= enet.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_enet = r2_score(test_y, pred_test)

# print(enet)
# print("r^2 on test data : %f" % r2_score_enet)

model_r2_score = r2_score_enet

model_analysis(enet,tag,model_r2_score)


#Unlike most other scores, R^2 score may be negative (it need not actually be the square of a quantity R).

print('\n'*1)

print ('Model formula : lprice ~ y + carat + clarity + color + depth + cut + table')


model_output

#predictions_list


# In[ ]:

###let's see if these predicted value are statistally different for the Models


# In[94]:

(predictions_comparison.head(5))


# In[ ]:




# In[28]:

round(0.00094373230442450738,5)


# In[ ]:

####Let did a little deeper


# In[90]:



def model_analysis (model,tag,model_r2_score):


    mae = metrics.mean_absolute_error(test_y, predictions)
    mse = metrics.mean_squared_error(test_y, predictions)


    pred_len=int(len(predictions))

    output_columns = ['Model','r^2 on test data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']
 
    model_output.loc[model_output.shape[0]] = (tag,'{0:.6f}'.format(model_r2_score),'{0:.6f}'.format(model.score(test_X, test_y)),
                        '{0:.6f}'.format(model.score(train_X, train_y)),np.sqrt(metrics.mean_squared_error(test_y,predictions)),(mae),mse)
    
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    my_title=('Prediction for Model: %s'%(model))
    plt.title(my_title +'\n')
    #plt.title('Prediction')
    plt.scatter(test_y, pred_test, 
         label='Prediction Accuracy')
    plt.legend(loc='upper left')
    #plt.xlabel('Actual')
    #plt.ylabel('Predicted')

    plt.show()

    # Release memory.
    plt.clf()
    plt.close()
    
    #Result obtained after running the algo. Comment the below two lines if you want to run the algo
    mae_list.append(mae)
    comb.append(tag)  

    
def run_ols_build(label,features):
    df = diamonds_train_R
    result = sm.ols(formula=label + "~" + build_features, data=df).fit()
    print result.params
    print result.summary()
    print("\n"*2)
    
    
    
# RFE gives us the following
# [(1.0, 'y'), 
#  (2.0, 'carat'), 
#  (3.0, 'z'), 
#  (4.0, 'x'), 
#  (5.0, 'clarity'), 
#  (6.0, 'color'), 
#  (7.0, 'depth'), 
#  (8.0, 'cut'), 
#  (9.0, 'table')]

from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
#Build Linear Models Using OLS
#from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm



train_X=diamonds_train_R[['y','clarity', 'color', 'depth',  'cut',  'table']]# taking the training data features
train_y=diamonds_train_R[['lprice']] # output of our training data
test_X =diamonds_test_R[['y','clarity', 'color', 'depth',  'cut',  'table']] # taking test data features
test_y=diamonds_test_R[['lprice']]  #output value of test data

# train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
# train_y=diamonds_train[['lprice']] # output of our training data
# test_X =diamonds_test[['carat','cut','clarity','color','x','y','z','depth','table']] # taking test data features
# test_y=diamonds_test[['lprice']]  #output value of test data

    

#List of combinations
comb = []

#Dictionary to store the MAE for all algorithms 
mae_list = []

output_columns = ['Model','r^2 on test data', 'Model Score on Training Data','Variance score','Root Mean Squared Error','Mean Absolute Error','Mean Squared Error']
model_output = pd.DataFrame(columns=output_columns)

# print(model_output.head())


alpha = 1.0
linear = LinearRegression()
tag = 'linear ~ y'

pred_test = linear.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_linear = r2_score(test_y, pred_test)

# print(linear)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_linear)

model_r2_score = r2_score_linear

print('\n'*1)

print ('Model formula : lprice ~ y  + clarity + color + depth + cut + table')

model_analysis(linear,tag,model_r2_score)

print('\n'*1)

train_X=diamonds_train_R[['carat','clarity', 'color', 'depth',  'cut',  'table']]# taking the training data features
train_y=diamonds_train_R[['lprice']] # output of our training data
test_X =diamonds_test_R[['carat','clarity', 'color', 'depth',  'cut',  'table']] # taking test data features
test_y=diamonds_test_R[['lprice']]  #output value of test data


alpha = 1.0
linear = LinearRegression()
tag = 'linear ~ carat'

pred_test = linear.fit(train_X, train_y).predict(test_X)
predictions = pred_test
r2_score_linear = r2_score(test_y, pred_test)

# print(linear)
# print ('alpha :' , alpha)
# print("r^2 on test data : %f" % r2_score_linear)

model_r2_score = r2_score_linear

print('\n'*1)

print ('Model formula : lprice ~ carat  + clarity + color + depth + cut + table')

model_analysis(linear,tag,model_r2_score)

print('\n'*1)






model_output.head()


#GradientBoostRegressor
#['y' 'clarity' 'z' 'carat' 'color' 'x' 'cut' 'depth' 'table']

#Recursive Feature Elimination top 5 predictors
#[(1.0, 'carat'), (1.0, 'clarity'), (1.0, 'x'), (1.0, 'y'), (1.0, 'z'), (2.0, 'color'), (3.0, 'depth'), (4.0, 'cut'), (5.0, 'table')]

#Ridge
# {'carat': 1.0, 
#  'x': 0.49, 
#  'y': 0.57, 
#  'z': 0.42}
#  'clarity': 0.06,
#  'color': 0.06,
#  'cut': 0.0,
#  'depth': 0.01,
#  'table': 0.0,


#y, X = dmatrices(label + '~' + features,

#names = list(train_X)


### Use y in OLS

train_X=diamonds_train_R[['y','clarity', 'color', 'depth',  'cut',  'table']]# taking the training data features
train_y=diamonds_train_R[['lprice']] # output of our training data
test_X =diamonds_test_R[['y','clarity', 'color', 'depth',  'cut',  'table']] # taking test data features
test_y=diamonds_test_R[['lprice']]  #output value of test data

build_features = "+".join(train_X.columns  - ['carat'])
#Pick out label                   
label = "".join(train_y.columns)
run_ols_build(label,build_features)


###### Use Carat in OLS

train_X=diamonds_train_R[['carat','clarity', 'color', 'depth',  'cut',  'table']]# taking the training data features
train_y=diamonds_train_R[['lprice']] # output of our training data
test_X =diamonds_test_R[['carat','clarity', 'color', 'depth',  'cut',  'table']] # taking test data features
test_y=diamonds_test_R[['lprice']]  #output value of test data


build_features = "+".join(train_X.columns  - ['y'])
#Pick out label                   
label = "".join(train_y.columns)
run_ols_build(label,build_features)    





# df = diamonds
# result = sm.ols(formula=label + "~" + build_features, data=df).fit()

#rank all features, i.e continue the elimination until the last one
#rfe = RFE(lr, n_features_to_select=1)
#rfe.fit(train_X,train_y)
 
#print "Features sorted by their rank:"
#print sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names))


print('\n'*1)


model_output.head()





# In[61]:


def run_ols_build(label,features):

    df = diamonds_train_R
    result = sm.ols(formula=label + "~" + build_features, data=df).fit()
    print result.params
    print result.summary()
    print("\n"*2)

    from statsmodels.graphics.gofplots import ProbPlot
    import seaborn as sns
    #plt.style.use('seaborn') # pretty matplotlib plots
    plt.rc('font', size=14)
    plt.rc('figure', titlesize=18)
    plt.rc('axes', labelsize=15)
    plt.rc('axes', titlesize=18)


    ## https://medium.com/@emredjan/emulating-r-regression-plots-in-python-43741952c034
    model_fit = result
    # fitted values (need a constant term for intercept)
    model_fitted_y = model_fit.fittedvalues

    # model residuals
    model_residuals = model_fit.resid

    # normalized residuals
    model_norm_residuals = model_fit.get_influence().resid_studentized_internal

    # absolute squared normalized residuals
    model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

    # absolute residuals
    model_abs_resid = np.abs(model_residuals)

    # leverage, from statsmodels internals
    model_leverage = model_fit.get_influence().hat_matrix_diag

    # cook's distance, from statsmodels internals
    model_cooks = model_fit.get_influence().cooks_distance[0]

    #Residual plot

    plot_lm_1 = plt.figure(1)
    plot_lm_1.set_figheight(8)
    plot_lm_1.set_figwidth(12)

    plot_lm_1.axes[0] = sns.residplot(model_fitted_y, 'lprice', data=diamonds_train_R, 
                              lowess=True, 
                              scatter_kws={'alpha': 0.5}, 
                              line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

    plot_lm_1.axes[0].set_title('Residuals vs Fitted')
    plot_lm_1.axes[0].set_xlabel('Fitted values')
    plot_lm_1.axes[0].set_ylabel('Residuals')

    # annotations
    abs_resid = model_abs_resid.sort_values(ascending=False)
    abs_resid_top_3 = abs_resid[:3]

    for i in abs_resid_top_3.index:
        plot_lm_1.axes[0].annotate(i, 
                                   xy=(model_fitted_y[i], 
                                       model_residuals[i]));


    
    plt.show() 
    
#         QQ = ProbPlot(model_norm_residuals)
#         plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)

#         plot_lm_2.set_figheight(8)
#         plot_lm_2.set_figwidth(12)

#         plot_lm_2.axes[0].set_title('Normal Q-Q')
#         plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
#         plot_lm_2.axes[0].set_ylabel('Standardized Residuals');

#         # annotations
#         #abs_norm_resid = np.fliplr(np.argsort(np.abs(model_norm_residuals)), 0)
#         abs_norm_resid = np.fliplr(np.argsort(np.abs(model_norm_residuals)))
#         abs_norm_resid_top_3 = abs_norm_resid[:3]

#         for r, i in enumerate(abs_norm_resid_top_3):
#             plot_lm_2.axes[0].annotate(i, 
#                                        xy=(np.fliplr(QQ.theoretical_quantiles, 0)[r],
#                                            model_norm_residuals[i]));

        
#         plt.show()   
    
#         #Scale-Location Plot    
#         plot_lm_3 = plt.figure(3)
#         plot_lm_3.set_figheight(8)
#         plot_lm_3.set_figwidth(12)

#         plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5)
#         sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt, 
#                     scatter=False, 
#                     ci=False, 
#                     lowess=True,
#                     line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

#         plot_lm_3.axes[0].set_title('Scale-Location')
#         plot_lm_3.axes[0].set_xlabel('Fitted values')
#         plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$');

#         # annotations
#         abs_sq_norm_resid = np.flip(np.argsort(model_norm_residuals_abs_sqrt), 0)
#         abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]

#         for i in abs_norm_resid_top_3:
#             plot_lm_3.axes[0].annotate(i, 
#                                        xy=(model_fitted_y[i], 
#                                            model_norm_residuals_abs_sqrt[i]));    

        
#         plot_lm_4 = plt.figure(4)
#         plot_lm_4.set_figheight(8)
#         plot_lm_4.set_figwidth(12)

#         plt.scatter(model_leverage, model_norm_residuals, alpha=0.5)
#         sns.regplot(model_leverage, model_norm_residuals, 
#                     scatter=False, 
#                     ci=False, 
#                     lowess=True,
#                     line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})


#         #Leverage Plot
#         plot_lm_4.axes[0].set_xlim(0, 0.20)
#         plot_lm_4.axes[0].set_ylim(-3, 5)
#         plot_lm_4.axes[0].set_title('Residuals vs Leverage')
#         plot_lm_4.axes[0].set_xlabel('Leverage')
#         plot_lm_4.axes[0].set_ylabel('Standardized Residuals')

#         # annotations
#         leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]

#         for i in leverage_top_3:
#             plot_lm_4.axes[0].annotate(i, 
#                                        xy=(model_leverage[i], 
#                                            model_norm_residuals[i]))

#         # shenanigans for cook's distance contours
#         def graph(formula, x_range, label=None):
#             x = x_range
#             y = formula(x)
#             plt.plot(x, y, label=label, lw=1, ls='--', color='red')

#         p = len(model_fit.params) # number of model parameters

#         graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x), 
#               np.linspace(0.001, 0.200, 50), 
#               'Cook\'s distance') # 0.5 line

#         graph(lambda x: np.sqrt((1 * p * (1 - x)) / x), 
#               np.linspace(0.001, 0.200, 50)) # 1 line

#         plt.legend(loc='upper right');
    
        
# RFE gives us the following
# [(1.0, 'y'), 
#  (2.0, 'carat'), 
#  (3.0, 'z'), 
#  (4.0, 'x'), 
#  (5.0, 'clarity'), 
#  (6.0, 'color'), 
#  (7.0, 'depth'), 
#  (8.0, 'cut'), 
#  (9.0, 'table')]

from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
#Build Linear Models Using OLS
#from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm


#GradientBoostRegressor
#['y' 'clarity' 'z' 'carat' 'color' 'x' 'cut' 'depth' 'table']

#Recursive Feature Elimination top 5 predictors
#[(1.0, 'carat'), (1.0, 'clarity'), (1.0, 'x'), (1.0, 'y'), (1.0, 'z'), (2.0, 'color'), (3.0, 'depth'), (4.0, 'cut'), (5.0, 'table')]

#Ridge
# {'carat': 1.0, 
#  'x': 0.49, 
#  'y': 0.57, 
#  'z': 0.42}
#  'clarity': 0.06,
#  'color': 0.06,
#  'cut': 0.0,
#  'depth': 0.01,
#  'table': 0.0,



### Use y in OLS

train_X=diamonds_train_R[['y','clarity', 'color', 'depth',  'cut',  'table']]# taking the training data features
train_y=diamonds_train_R[['lprice']] # output of our training data
test_X =diamonds_test_R[['y','clarity', 'color', 'depth',  'cut',  'table']] # taking test data features
test_y=diamonds_test_R[['lprice']]  #output value of test data

build_features = "+".join(train_X.columns  - ['carat'])
#Pick out label                   
label = "".join(train_y.columns)
run_ols_build(label,build_features)


###### Use Carat in OLS

train_X=diamonds_train_R[['carat','clarity', 'color', 'depth',  'cut',  'table']]# taking the training data features
train_y=diamonds_train_R[['lprice']] # output of our training data
test_X =diamonds_test_R[['carat','clarity', 'color', 'depth',  'cut',  'table']] # taking test data features
test_y=diamonds_test_R[['lprice']]  #output value of test data


build_features = "+".join(train_X.columns  - ['y'])
#Pick out label                   
label = "".join(train_y.columns)
run_ols_build(label,build_features)    





# In[62]:

def run_ols_build(label,features):
    
        df = diamonds_train_R
        result = sm.ols(formula=label + "~" + build_features, data=df).fit()
        print result.params
        print result.summary()
        print("\n"*2)

        from statsmodels.graphics.gofplots import ProbPlot
        import seaborn as sns
        #plt.style.use('seaborn') # pretty matplotlib plots
        plt.rc('font', size=14)
        plt.rc('figure', titlesize=18)
        plt.rc('axes', labelsize=15)
        plt.rc('axes', titlesize=18)

    
        ## https://medium.com/@emredjan/emulating-r-regression-plots-in-python-43741952c034
        model_fit = result
        # fitted values (need a constant term for intercept)
        model_fitted_y = model_fit.fittedvalues

        # model residuals
        model_residuals = model_fit.resid

        # normalized residuals
        model_norm_residuals = model_fit.get_influence().resid_studentized_internal

        # absolute squared normalized residuals
        model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

        # absolute residuals
        model_abs_resid = np.abs(model_residuals)

        # leverage, from statsmodels internals
        model_leverage = model_fit.get_influence().hat_matrix_diag

        # cook's distance, from statsmodels internals
        model_cooks = model_fit.get_influence().cooks_distance[0]


        
        QQ = ProbPlot(model_norm_residuals)
        plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)

        plot_lm_2.set_figheight(8)
        plot_lm_2.set_figwidth(12)

        plot_lm_2.axes[0].set_title('Normal Q-Q')
        plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
        plot_lm_2.axes[0].set_ylabel('Standardized Residuals');

        # annotations
        #abs_norm_resid = np.fliplr(np.argsort(np.abs(model_norm_residuals)), 0)
        abs_norm_resid = np.fliplr(np.argsort(np.abs(model_norm_residuals)))
        abs_norm_resid_top_3 = abs_norm_resid[:3]

        for r, i in enumerate(abs_norm_resid_top_3):
            plot_lm_2.axes[0].annotate(i, 
                                       xy=(np.fliplr(QQ.theoretical_quantiles, 0)[r],
                                           model_norm_residuals[i]));

            
        plt.show()   

            
# RFE gives us the following
# [(1.0, 'y'), 
#  (2.0, 'carat'), 
#  (3.0, 'z'), 
#  (4.0, 'x'), 
#  (5.0, 'clarity'), 
#  (6.0, 'color'), 
#  (7.0, 'depth'), 
#  (8.0, 'cut'), 
#  (9.0, 'table')]

from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
#Build Linear Models Using OLS
#from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm


#GradientBoostRegressor
#['y' 'clarity' 'z' 'carat' 'color' 'x' 'cut' 'depth' 'table']

#Recursive Feature Elimination top 5 predictors
#[(1.0, 'carat'), (1.0, 'clarity'), (1.0, 'x'), (1.0, 'y'), (1.0, 'z'), (2.0, 'color'), (3.0, 'depth'), (4.0, 'cut'), (5.0, 'table')]

#Ridge
# {'carat': 1.0, 
#  'x': 0.49, 
#  'y': 0.57, 
#  'z': 0.42}
#  'clarity': 0.06,
#  'color': 0.06,
#  'cut': 0.0,
#  'depth': 0.01,
#  'table': 0.0,



### Use y in OLS

train_X=diamonds_train_R[['y','clarity', 'color', 'depth',  'cut',  'table']]# taking the training data features
train_y=diamonds_train_R[['lprice']] # output of our training data
test_X =diamonds_test_R[['y','clarity', 'color', 'depth',  'cut',  'table']] # taking test data features
test_y=diamonds_test_R[['lprice']]  #output value of test data

build_features = "+".join(train_X.columns  - ['carat'])
#Pick out label                   
label = "".join(train_y.columns)
run_ols_build(label,build_features)


###### Use Carat in OLS

train_X=diamonds_train_R[['carat','clarity', 'color', 'depth',  'cut',  'table']]# taking the training data features
train_y=diamonds_train_R[['lprice']] # output of our training data
test_X =diamonds_test_R[['carat','clarity', 'color', 'depth',  'cut',  'table']] # taking test data features
test_y=diamonds_test_R[['lprice']]  #output value of test data


build_features = "+".join(train_X.columns  - ['y'])
#Pick out label                   
label = "".join(train_y.columns)
run_ols_build(label,build_features)    



# In[63]:

def run_ols_build(label,features):
    
        df = diamonds_train_R
        result = sm.ols(formula=label + "~" + build_features, data=df).fit()
        print result.params
        print result.summary()
        print("\n"*2)

        from statsmodels.graphics.gofplots import ProbPlot
        import seaborn as sns
        #plt.style.use('seaborn') # pretty matplotlib plots
        plt.rc('font', size=14)
        plt.rc('figure', titlesize=18)
        plt.rc('axes', labelsize=15)
        plt.rc('axes', titlesize=18)

    
        ## https://medium.com/@emredjan/emulating-r-regression-plots-in-python-43741952c034
        model_fit = result
        # fitted values (need a constant term for intercept)
        model_fitted_y = model_fit.fittedvalues

        # model residuals
        model_residuals = model_fit.resid

        # normalized residuals
        model_norm_residuals = model_fit.get_influence().resid_studentized_internal

        # absolute squared normalized residuals
        model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

        # absolute residuals
        model_abs_resid = np.abs(model_residuals)

        # leverage, from statsmodels internals
        model_leverage = model_fit.get_influence().hat_matrix_diag

        # cook's distance, from statsmodels internals
        model_cooks = model_fit.get_influence().cooks_distance[0]

      
   
        
        #Scale-Location Plot    
        plot_lm_3 = plt.figure(3)
        plot_lm_3.set_figheight(8)
        plot_lm_3.set_figwidth(12)

        plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5)
        sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt, 
                    scatter=False, 
                    ci=False, 
                    lowess=True,
                    line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

        plot_lm_3.axes[0].set_title('Scale-Location')
        plot_lm_3.axes[0].set_xlabel('Fitted values')
        plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$');

        # annotations
        abs_sq_norm_resid = np.fliplr(np.argsort(model_norm_residuals_abs_sqrt), 0)
        abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]

        for i in abs_norm_resid_top_3:
            plot_lm_3.axes[0].annotate(i, 
                                       xy=(model_fitted_y[i], 
                                           model_norm_residuals_abs_sqrt[i]));    

        plt.show()
            
# RFE gives us the following
# [(1.0, 'y'), 
#  (2.0, 'carat'), 
#  (3.0, 'z'), 
#  (4.0, 'x'), 
#  (5.0, 'clarity'), 
#  (6.0, 'color'), 
#  (7.0, 'depth'), 
#  (8.0, 'cut'), 
#  (9.0, 'table')]

from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
#Build Linear Models Using OLS
#from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm


#GradientBoostRegressor
#['y' 'clarity' 'z' 'carat' 'color' 'x' 'cut' 'depth' 'table']

#Recursive Feature Elimination top 5 predictors
#[(1.0, 'carat'), (1.0, 'clarity'), (1.0, 'x'), (1.0, 'y'), (1.0, 'z'), (2.0, 'color'), (3.0, 'depth'), (4.0, 'cut'), (5.0, 'table')]

#Ridge
# {'carat': 1.0, 
#  'x': 0.49, 
#  'y': 0.57, 
#  'z': 0.42}
#  'clarity': 0.06,
#  'color': 0.06,
#  'cut': 0.0,
#  'depth': 0.01,
#  'table': 0.0,



### Use y in OLS

train_X=diamonds_train_R[['y','clarity', 'color', 'depth',  'cut',  'table']]# taking the training data features
train_y=diamonds_train_R[['lprice']] # output of our training data
test_X =diamonds_test_R[['y','clarity', 'color', 'depth',  'cut',  'table']] # taking test data features
test_y=diamonds_test_R[['lprice']]  #output value of test data

build_features = "+".join(train_X.columns  - ['carat'])
#Pick out label                   
label = "".join(train_y.columns)
run_ols_build(label,build_features)


###### Use Carat in OLS

train_X=diamonds_train_R[['carat','clarity', 'color', 'depth',  'cut',  'table']]# taking the training data features
train_y=diamonds_train_R[['lprice']] # output of our training data
test_X =diamonds_test_R[['carat','clarity', 'color', 'depth',  'cut',  'table']] # taking test data features
test_y=diamonds_test_R[['lprice']]  #output value of test data


build_features = "+".join(train_X.columns  - ['y'])
#Pick out label                   
label = "".join(train_y.columns)
run_ols_build(label,build_features)    



# In[65]:

def run_ols_build(label,features):
    
        df = diamonds_train_R
        result = sm.ols(formula=label + "~" + build_features, data=df).fit()
        print result.params
        print result.summary()
        print("\n"*2)

        from statsmodels.graphics.gofplots import ProbPlot
        import seaborn as sns
        #plt.style.use('seaborn') # pretty matplotlib plots
        plt.rc('font', size=14)
        plt.rc('figure', titlesize=18)
        plt.rc('axes', labelsize=15)
        plt.rc('axes', titlesize=18)

    
        ## https://medium.com/@emredjan/emulating-r-regression-plots-in-python-43741952c034
        model_fit = result
        # fitted values (need a constant term for intercept)
        model_fitted_y = model_fit.fittedvalues

        # model residuals
        model_residuals = model_fit.resid

        # normalized residuals
        model_norm_residuals = model_fit.get_influence().resid_studentized_internal

        # absolute squared normalized residuals
        model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

        # absolute residuals
        model_abs_resid = np.abs(model_residuals)

        # leverage, from statsmodels internals
        model_leverage = model_fit.get_influence().hat_matrix_diag

        # cook's distance, from statsmodels internals
        model_cooks = model_fit.get_influence().cooks_distance[0]

        plot_lm_4 = plt.figure(4)
        plot_lm_4.set_figheight(8)
        plot_lm_4.set_figwidth(12)

        plt.scatter(model_leverage, model_norm_residuals, alpha=0.5)
        sns.regplot(model_leverage, model_norm_residuals, 
                    scatter=False, 
                    ci=False, 
                    lowess=True,
                    line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
        #Leverage Plot
        plot_lm_4.axes[0].set_xlim(0, 0.20)
        plot_lm_4.axes[0].set_ylim(-3, 5)
        plot_lm_4.axes[0].set_title('Residuals vs Leverage')
        plot_lm_4.axes[0].set_xlabel('Leverage')
        plot_lm_4.axes[0].set_ylabel('Standardized Residuals')

        # annotations
        leverage_top_3 = np.fliplr(np.argsort(model_cooks), 0)[:3]

        for i in leverage_top_3:
            plot_lm_4.axes[0].annotate(i, 
                                       xy=(model_leverage[i], 
                                           model_norm_residuals[i]))

        # shenanigans for cook's distance contours
        def graph(formula, x_range, label=None):
            x = x_range
            y = formula(x)
            plt.plot(x, y, label=label, lw=1, ls='--', color='red')

        p = len(model_fit.params) # number of model parameters

        graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x), 
              np.linspace(0.001, 0.200, 50), 
              'Cook\'s distance') # 0.5 line

        graph(lambda x: np.sqrt((1 * p * (1 - x)) / x), 
              np.linspace(0.001, 0.200, 50)) # 1 line

        plt.legend(loc='upper right');
        
        plt.show()    
# RFE gives us the following
# [(1.0, 'y'), 
#  (2.0, 'carat'), 
#  (3.0, 'z'), 
#  (4.0, 'x'), 
#  (5.0, 'clarity'), 
#  (6.0, 'color'), 
#  (7.0, 'depth'), 
#  (8.0, 'cut'), 
#  (9.0, 'table')]

from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
#Build Linear Models Using OLS
#from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm


#GradientBoostRegressor
#['y' 'clarity' 'z' 'carat' 'color' 'x' 'cut' 'depth' 'table']

#Recursive Feature Elimination top 5 predictors
#[(1.0, 'carat'), (1.0, 'clarity'), (1.0, 'x'), (1.0, 'y'), (1.0, 'z'), (2.0, 'color'), (3.0, 'depth'), (4.0, 'cut'), (5.0, 'table')]

#Ridge
# {'carat': 1.0, 
#  'x': 0.49, 
#  'y': 0.57, 
#  'z': 0.42}
#  'clarity': 0.06,
#  'color': 0.06,
#  'cut': 0.0,
#  'depth': 0.01,
#  'table': 0.0,



### Use y in OLS

train_X=diamonds_train_R[['y','clarity', 'color', 'depth',  'cut',  'table']]# taking the training data features
train_y=diamonds_train_R[['lprice']] # output of our training data
test_X =diamonds_test_R[['y','clarity', 'color', 'depth',  'cut',  'table']] # taking test data features
test_y=diamonds_test_R[['lprice']]  #output value of test data

build_features = "+".join(train_X.columns  - ['carat'])
#Pick out label                   
label = "".join(train_y.columns)
run_ols_build(label,build_features)


###### Use Carat in OLS

train_X=diamonds_train_R[['carat','clarity', 'color', 'depth',  'cut',  'table']]# taking the training data features
train_y=diamonds_train_R[['lprice']] # output of our training data
test_X =diamonds_test_R[['carat','clarity', 'color', 'depth',  'cut',  'table']] # taking test data features
test_y=diamonds_test_R[['lprice']]  #output value of test data


build_features = "+".join(train_X.columns  - ['y'])
#Pick out label                   
label = "".join(train_y.columns)
run_ols_build(label,build_features)    



# In[53]:

build_features, label,train_X.columns


# # Lasso Cross validation - determining alpha

# In[63]:

from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
train_y=diamonds_train[['lprice']] # output of our training data
test_X =diamonds_test[['carat','cut','clarity','color','x','y','z','depth','table']] # taking test data features
test_y=diamonds_test[['lprice']]  #output value of test data



lasso = Lasso(random_state=0)
alphas = np.logspace(-4, -0.5, 30)

scores = list()
scores_std = list()

n_folds = 3

for alpha in alphas:
    lasso.alpha = alpha
    this_scores = cross_val_score(lasso, train_X, train_y, cv=n_folds, n_jobs=1)
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))

scores, scores_std = np.array(scores), np.array(scores_std)

plt.figure().set_size_inches(8, 6)
plt.semilogx(alphas, scores)

# plot error lines showing +/- std. errors of the scores
std_error = scores_std / np.sqrt(n_folds)

plt.semilogx(alphas, scores + std_error, 'b--')
plt.semilogx(alphas, scores - std_error, 'b--')

# alpha=0.2 controls the translucency of the fill color
plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

plt.ylabel('CV score +/- std error')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.xlim([alphas[0], alphas[-1]])

plt.show()


# Release memory.
plt.clf()
plt.close()


# LassoCV object that sets its alpha parameter automatically 
# from the data by internal cross-validation
# # performs cross-validation on the training data it receives).
# We use external cross-validation to see how much the automatically obtained
# alphas differ across different cross-validation folds.
lasso_cv = LassoCV(alphas=alphas, random_state=0)
k_fold = KFold(100)


X = np.array(train_X)
y = np.array(train_y)

#Selection of Alpha

for k, (train, test) in enumerate(k_fold.split(X, y)):
    #print(train,test)
    lasso_cv.fit(X[train], y[train])
    print("[fold {0}] alpha: {1:.5f}, score: {2:.5f}".
          format(k, lasso_cv.alpha_, lasso_cv.score(X[test], y[test])))
print()
print("Compare the  alphas obtainedfor the")
print("subsets of the data and moreover, and see if the scores differ")
print("substantially.")

plt.show()
print(lasso_cv.alpha_)
print("Done")


# ### Lasso alpha determined by Cross Validation

# In[58]:

from sklearn import linear_model, datasets

lasso = linear_model.LassoCV()

lasso.fit(train_X, train_y)
LassoCV(alphas=None, copy_X=True, cv=None, eps=0.001, fit_intercept=True,
    max_iter=1000, n_alphas=100, n_jobs=1, normalize=False, positive=False,
    precompute='auto', random_state=None, selection='cyclic', tol=0.0001,
    verbose=False)
 # The estimator chose automatically its lambda:
lasso.alpha_ 


# In[34]:

round(0.00047489749072328851,5)


# In[ ]:

#Determine Alpha on only 1 of quartet


# In[7]:

from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


train_X=diamonds_train[['cut','clarity','color','y','depth','table']]# taking the training data features
train_y=diamonds_train[['lprice']] # output of our training data
test_X =diamonds_test[['cut','clarity','color','y','depth','table']] # taking test data features
test_y=diamonds_test[['lprice']]  #output value of test data



lasso = Lasso(random_state=0)
alphas = np.logspace(-4, -0.5, 30)

scores = list()
scores_std = list()

n_folds = 3

for alpha in alphas:
    lasso.alpha = alpha
    this_scores = cross_val_score(lasso, train_X, train_y, cv=n_folds, n_jobs=1)
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))

scores, scores_std = np.array(scores), np.array(scores_std)

plt.figure().set_size_inches(8, 6)
plt.semilogx(alphas, scores)

# plot error lines showing +/- std. errors of the scores
std_error = scores_std / np.sqrt(n_folds)

plt.semilogx(alphas, scores + std_error, 'b--')
plt.semilogx(alphas, scores - std_error, 'b--')

# alpha=0.2 controls the translucency of the fill color
plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

plt.ylabel('CV score +/- std error')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.xlim([alphas[0], alphas[-1]])

plt.show()


# Release memory.
plt.clf()
plt.close()


# LassoCV object that sets its alpha parameter automatically 
# from the data by internal cross-validation
# # performs cross-validation on the training data it receives).
# We use external cross-validation to see how much the automatically obtained
# alphas differ across different cross-validation folds.
lasso_cv = LassoCV(alphas=alphas, random_state=0)
k_fold = KFold(100)


X = np.array(train_X)
y = np.array(train_y)

#Selection of Alpha

for k, (train, test) in enumerate(k_fold.split(X, y)):
    #print(train,test)
    lasso_cv.fit(X[train], y[train])
    print("[fold {0}] alpha: {1:.5f}, score: {2:.5f}".
          format(k, lasso_cv.alpha_, lasso_cv.score(X[test], y[test])))
print()
print("Compare the  alphas obtained for the")
print("subsets of the data and moreover, and see if the scores differ")
print("substantially.")

plt.show()
print(lasso_cv.alpha_)
print("Done")


# ### Lasso alpha determined by Cross Validation

# In[8]:

from sklearn import linear_model, datasets

lasso = linear_model.LassoCV()

lasso.fit(train_X, train_y)
LassoCV(alphas=None, copy_X=True, cv=None, eps=0.001, fit_intercept=True,
    max_iter=1000, n_alphas=100, n_jobs=1, normalize=False, positive=False,
    precompute='auto', random_state=None, selection='cyclic', tol=0.0001,
    verbose=False)
 # The estimator chose automatically its lambda:
lasso.alpha_ 


# ### ElaticNet CV alpha

# In[ ]:

#ElasticNet(alpha=alpha, l1_ratio=0.7)


# In[11]:

from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV

train_X=diamonds_train[['cut','clarity','color','y','depth','table']]# taking the training data features
train_y=diamonds_train[['lprice']] # output of our training data
test_X =diamonds_test[['cut','clarity','color','y','depth','table']] # taking test data features
test_y=diamonds_test[['lprice']]  #output value of test data


enet = ElasticNet(l1_ratio=0.7)
#lasso = Lasso(random_state=0)
alphas = np.logspace(-4, -0.5, 30)

scores = list()
scores_std = list()

n_folds = 3

for alpha in alphas:
    enet.alpha = alpha
    this_scores = cross_val_score(enet, train_X, train_y, cv=n_folds, n_jobs=1)
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))

scores, scores_std = np.array(scores), np.array(scores_std)

plt.figure().set_size_inches(8, 6)
plt.semilogx(alphas, scores)

# plot error lines showing +/- std. errors of the scores
std_error = scores_std / np.sqrt(n_folds)

plt.semilogx(alphas, scores + std_error, 'b--')
plt.semilogx(alphas, scores - std_error, 'b--')

# alpha=0.2 controls the translucency of the fill color
plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

plt.ylabel('CV score +/- std error')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.xlim([alphas[0], alphas[-1]])

plt.show()


# Release memory.
plt.clf()
plt.close()


# LassoCV object that sets its alpha parameter automatically 
# from the data by internal cross-validation
# # performs cross-validation on the training data it receives).
# We use external cross-validation to see how much the automatically obtained
# alphas differ across different cross-validation folds.
enet_cv = ElasticNetCV(alphas=alphas, random_state=0)
k_fold = KFold(100)


X = np.array(train_X)
y = np.array(train_y)

#Selection of Alpha

for k, (train, test) in enumerate(k_fold.split(X, y)):
    #print(train,test)
    enet_cv.fit(X[train], y[train])
    print("[fold {0}] alpha: {1:.5f}, score: {2:.5f}".
          format(k, enet_cv.alpha_, enet_cv.score(X[test], y[test])))
print()
print("Compare the  alphas obtained for the")
print("subsets of the data and moreover, and see if the scores differ")
print("substantially.")

plt.show()
print(enet_cv.alpha_)
print("Done")


# In[12]:

from sklearn import linear_model, datasets

enet = linear_model.ElasticNetCV()

enet.fit(train_X, train_y)
ElasticNetCV(alphas=None, copy_X=True, cv=None, eps=0.001, fit_intercept=True,
    max_iter=1000, n_alphas=100, n_jobs=1, normalize=False, positive=False,
    precompute='auto', random_state=None, selection='cyclic', tol=0.0001,
    verbose=False)
 # The estimator chose automatically its lambda:
enet.alpha_ 


# In[33]:

round(0.00094373230442450738,5)


# In[ ]:

#Try the CV using all feature to see if we detect any difference


# In[21]:

from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV

train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
train_y=diamonds_train[['lprice']] # output of our training data
test_X =diamonds_test[['carat','cut','clarity','color','x','y','z','depth','table']] # taking test data features
test_y=diamonds_test[['lprice']]  #output value of test data


enet = ElasticNet(l1_ratio=0.7)
#lasso = Lasso(random_state=0)
alphas = np.logspace(-4, -0.5, 30)

scores = list()
scores_std = list()

n_folds = 3

for alpha in alphas:
    enet.alpha = alpha
    this_scores = cross_val_score(enet, train_X, train_y, cv=n_folds, n_jobs=1)
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))

scores, scores_std = np.array(scores), np.array(scores_std)

plt.figure().set_size_inches(8, 6)
plt.semilogx(alphas, scores)

# plot error lines showing +/- std. errors of the scores
std_error = scores_std / np.sqrt(n_folds)

plt.semilogx(alphas, scores + std_error, 'b--')
plt.semilogx(alphas, scores - std_error, 'b--')

# alpha=0.2 controls the translucency of the fill color
plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

plt.ylabel('CV score +/- std error')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.xlim([alphas[0], alphas[-1]])

plt.show()


# Release memory.
plt.clf()
plt.close()


# LassoCV object that sets its alpha parameter automatically 
# from the data by internal cross-validation
# # performs cross-validation on the training data it receives).
# We use external cross-validation to see how much the automatically obtained
# alphas differ across different cross-validation folds.
enet_cv = ElasticNetCV(alphas=alphas, random_state=0)
k_fold = KFold(100)


X = np.array(train_X)
y = np.array(train_y)

#Selection of Alpha

for k, (train, test) in enumerate(k_fold.split(X, y)):
    #print(train,test)
    enet_cv.fit(X[train], y[train])
    print("[fold {0}] alpha: {1:.5f}, score: {2:.5f}".
          format(k, enet_cv.alpha_, enet_cv.score(X[test], y[test])))
print()
print("Compare the  alphas obtained for the")
print("subsets of the data and moreover, and see if the scores differ")
print("substantially.")

plt.show()
print(enet_cv.alpha_)
print("Done")


# In[22]:

from sklearn import linear_model, datasets

enet = linear_model.ElasticNetCV()

enet.fit(train_X, train_y)
ElasticNetCV(alphas=None, copy_X=True, cv=None, eps=0.001, fit_intercept=True,
    max_iter=1000, n_alphas=100, n_jobs=1, normalize=False, positive=False,
    precompute='auto', random_state=None, selection='cyclic', tol=0.0001,
    verbose=False)
 # The estimator chose automatically its lambda:
enet.alpha_ 


# In[ ]:




# In[197]:

for  k, (train, test) in enumerate(k_fold.split(train_X, train_y)):
    print(k,train,test)


# In[90]:

clf.coef_,ols.coef_


# ### BaysianRidge OLS Linear Regression

# In[120]:

from sklearn.linear_model import BayesianRidge, LinearRegression

train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
train_y=diamonds_train[['lprice']] # output of our training data
test_X =diamonds_test[['carat','cut','clarity','color','x','y','z','depth','table']] # taking test data features
test_y=diamonds_test[['lprice']]  #output value of test data


clf = BayesianRidge(compute_score=True)
clf.fit(train_X, train_y)

ols = LinearRegression()
ols.fit(train_X, train_y)


lw = 2 #linewidth

# Release memory.
plt.clf()
plt.close()

plt.figure(figsize=(6, 5))
plt.title("Weights of the model")
plt.plot(clf.coef_, color='lightgreen', linewidth=lw,
         label="Bayesian Ridge estimate")
#plt.plot(w, color='gold', linewidth=lw, label="Ground truth")
plt.plot(ols.coef_, color='navy',linestyle='--', label="OLS estimate")
plt.xlabel("Features")
plt.ylabel("Values of the weights")
plt.legend(loc="best", prop=dict(size=12))

plt.show()

# Release memory.
plt.clf()
plt.close()




# In[206]:

train_X.head()


# In[101]:

n_features = sfm.transform(X).shape[1]
print(n_features)


# In[4]:

import matplotlib.pyplot as plt
#from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression

from sklearn.svm import SVC

# Build a classification task using 3 informative features
# X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
#                            n_redundant=2, n_repeated=0, n_classes=8,
#                            n_clusters_per_class=1, random_state=0)

X = np.array(train_X)
y = np.array(train_y)

#Reshape out Array from (37741L, 1L)) to 37741L,
y = np.reshape(y, -1)


names = list(train_X)
 
#use linear regression as the model
#lr = LinearRegression()
#rank all features, i.e continue the elimination until the last one
#rfe = RFE(lr, n_features_to_select=1)
#rfe.fit(train_X,train_y)
 

# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()




# In[121]:

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification


# Build a classification task using 3 informative features
# X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
#                            n_redundant=2, n_repeated=0, n_classes=8,
#                            n_clusters_per_class=1, random_state=0)

X = np.array(train_X)
y = np.array(train_y)

#Reshape out Array from (37741L, 1L)) to 37741L,
y = np.reshape(y, -1)

# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


# # Feature selection using SelectFromModel and LassoCV - index out of bounds - can't get the feature2

# In[50]:

#We'll need a normalised dataset later on so lets concatenate test and train  here

from sklearn.utils import shuffle
diamonds_norm = None
diamonds_norm = pd.concat([diamonds_train, diamonds_test], ignore_index=True)
diamonds_target=diamonds_norm['lprice'] # #Give diamonds.target the target value lprice

#Let indexes_to_drop be an array of positional indexes to drop ([0,1,2,3,4,5,6,7,8,9,11] in our case everything but lprice).
diamonds_target_use = None

diamonds_target_use = diamonds_norm.iloc[:,[10]] #lprice

#We don't want to  use the idela_flag or the price so lets drop them
diamonds_norm_use = None

#Let indexes_to_drop be an array of positional indexes to drop ([6,11] in our case).
diamonds_norm_use = diamonds_norm.iloc[:,[0,1,2,3,4,5,7,8,9,10]] # All out other columns excluding price and ideal_flag
diamonds_norm_use = diamonds_norm.iloc[:,[0,1,2,3,4,5,7,8,9]] # All out other columns excluding lprice



X, y = shuffle(diamonds_norm_use, diamonds_target_use, random_state=23)
X = X.astype(np.float32)



# offset = int(X.shape[0] * 0.9)
# X_train, y_train = X[:offset], y[:offset]
# X_test, y_test = X[offset:], y[offset:]

# names = list(X_train)


# X_train = np.array(X_train)
# y_train = np.array(y_train)

# X_test = np.array(X_test)
# y_test = np.array(y_test)


#Reshape out Array from (37741L, 1L)) to 37741L,
# y_train = np.reshape(y_train, -1)
# y_test = np.reshape(y_test, -1)


X = np.array(X)
y = np.array(y)
y = np.reshape(y, -1)

# We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
clf = LassoCV()

###Issue is witht he sfm.transform the array returned is 1L and should be 2L

# Set a minimum threshold of 0.25
sfm = SelectFromModel(clf, threshold=0.25)
sfm.fit(X, y)
n_features = sfm.transform(X).shape[1]

# Reset the threshold till the number of features equals two.
# Note that the attribute can be set directly instead of repeatedly
# fitting the metatransformer.
while n_features > 2:
    sfm.threshold += 0.1
    X_transform = sfm.transform(X)
    n_features = X_transform.shape[1]

# Plot the selected two features from X.
plt.title(
    "Features selected from Diamonds using SelectFromModel with "
    "threshold %0.3f." % sfm.threshold)
feature1 = X_transform[:, 0]
feature2 = X_transform[:, 1]
feature3 = X_transform[:, 2]
plt.plot(feature1, feature2, 'r.')
plt.xlabel("Feature number 1")
plt.ylabel("Feature number 2")
plt.ylim([np.min(feature2), np.max(feature2)])
plt.show()


# In[48]:

#Credit # Author: Manoj Kumar <mks542@nyu.edu>
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np

#from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
train_y=diamonds_train[['lprice']] # output of our training data
test_X =diamonds_test[['carat','cut','clarity','color','x','y','z','depth','table']] # taking test data features
test_y=diamonds_test[['lprice']]  #output value of test data

X = np.array(train_X)
y = np.array(train_y)


#Just check data
df_temp = pd.DataFrame(X)
df_temp2 = pd.DataFrame(y)

#Reshape out Array from (37741L, 1L)) to 37741L,
y = np.reshape(y, -1)

#alpha = 0.1
# We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
clf = LassoCV()

###Issue is with the sfm.transform the array returned is 1L and should be 2L
# Set a minimum threshold of 0.25

sfm = SelectFromModel(clf, threshold=0.25)
sfm.fit(X, y)
sfm.fit(train_X, train_y)
n_features = sfm.transform(X).shape[1]

# Reset the threshold till the number of features equals two.
# Note that the attribute can be set directly instead of repeatedly
# fitting the metatransformer.
while n_features > 2:
    sfm.threshold += 0.1
    X_transform = sfm.transform(X)
    n_features = X_transform.shape[1]

# Plot the selected two features from X.
plt.title(
    "Features selected from Diamonds using SelectFromModel with "
    "threshold %0.3f." % sfm.threshold)
feature1 = X_transform[:, 0]
feature2 = X_transform[:, 1]
#feature3 = X_transform[:, 2]
plt.plot(feature1, feature2, 'r.')
plt.xlabel("Feature number 1")
plt.ylabel("Feature number 2")
plt.ylim([np.min(feature2), np.max(feature2)])
plt.show()


# In[52]:

#Credit # Author: Manoj Kumar <mks542@nyu.edu>
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np

#from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression

train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
train_y=diamonds_train[['lprice']] # output of our training data
test_X =diamonds_test[['carat','cut','clarity','color','x','y','z','depth','table']] # taking test data features
test_y=diamonds_test[['lprice']]  #output value of test data

X = np.array(train_X)
y = np.array(train_y)


#Just check data
df_temp = pd.DataFrame(X)
df_temp2 = pd.DataFrame(y)

#Reshape out Array from (37741L, 1L)) to 37741L,
y = np.reshape(y, -1)

#alpha = 0.1
# We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
clf = LinearRegression()

###Issue is with the sfm.transform the array returned is 1L and should be 2L
# Set a minimum threshold of 0.25

sfm = SelectFromModel(clf, threshold=0.25)
sfm.fit(X, y)
sfm.fit(train_X, train_y)
n_features = sfm.transform(X).shape[1]

# Reset the threshold till the number of features equals two.
# Note that the attribute can be set directly instead of repeatedly
# fitting the metatransformer.
while n_features > 2:
    sfm.threshold += 0.1
    X_transform = sfm.transform(X)
    n_features = X_transform.shape[1]

# Plot the selected two features from X.
plt.title(
    "Features selected from Diamonds using SelectFromModel with "
    "threshold %0.3f." % sfm.threshold)
feature1 = X_transform[:, 0]
feature2 = X_transform[:, 1]
#feature3 = X_transform[:, 2]
plt.plot(feature1, feature2, 'r.')
plt.xlabel("Feature number 1")
plt.ylabel("Feature number 2")
plt.ylim([np.min(feature2), np.max(feature2)])
plt.show()


# In[54]:

df_temp3 = pd.DataFrame(feature1)


# In[147]:

my_y = np.reshape(y, -1)

my_y


# In[31]:

#import plotly.plotly as py
#import plotly.graph_objs as go

import numpy as np
from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

# Load the boston dataset.
boston = load_boston()
X, y = boston['data'], boston['target']


#Just check data
df_temp = pd.DataFrame(X)
df_temp2 = pd.DataFrame(y)
# We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
clf = LassoCV()

# Set a minimum threshold of 0.25
sfm = SelectFromModel(clf, threshold=0.25)
sfm.fit(X, y)
n_features = sfm.transform(X).shape[1]


# In[24]:

n_features,sfm.transform(X).shape[1],y.shape


# In[8]:

#import plotly.plotly as py
#import plotly.graph_objs as go

import numpy as np
from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

# Load the boston dataset.
boston = load_boston()
X, y = boston['data'], boston['target']

#Just check data
df_temp = pd.DataFrame(X)
df_temp2 = pd.DataFrame(y)
# We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
clf = LassoCV()

# Set a minimum threshold of 0.25
sfm = SelectFromModel(clf, threshold=0.25)
sfm.fit(X, y)
n_features = sfm.transform(X).shape[1]

# Reset the threshold till the number of features equals two.
# Note that the attribute can be set directly instead of repeatedly
# fitting the metatransformer.
while n_features > 2:
    sfm.threshold += 0.1
    X_transform = sfm.transform(X)
    n_features = X_transform.shape[1]
    
# Plot the selected two features from X.
plt.title(
    "Features selected from Boston using SelectFromModel with "
    "threshold %0.3f." % sfm.threshold)
feature1 = X_transform[:, 0]
feature2 = X_transform[:, 1]
plt.plot(feature1, feature2, 'r.')
plt.xlabel("Feature number 1")
plt.ylabel("Feature number 2")
plt.ylim([np.min(feature2), np.max(feature2)])
plt.show()


# In[19]:

X.shape,y.shape


# In[14]:

X.shape,y.shape,sfm,sfm.transform(X).shape[0],sfm.transform(X).shape[1],n_features


# # Principal Component Analysis

# In[3]:

#print(__doc__)


# Code source: Gaël Varoquaux
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn import decomposition
from sklearn import datasets

np.random.seed(5)

centers = [[1, 1], [-1, -1]]#, [1, -1]]

#We have 'cut' included but the target is ideal_flag...a predictor which define the ideal_flag
#We have dropped 'cut' as the target is ideal_flag which is based on cut value
train_X=diamonds_train[['carat','clarity','color','x','y','z','depth','table','lprice']]# taking the training data features
train_y=diamonds_train.ideal_flag# output of our training data
test_X=diamonds_test[['carat','clarity','color','x','y','z','depth','table','lprice']] # taking test data features
test_y=diamonds_test.ideal_flag   #output value of test data


X=np.array(train_X)
y=np.array(train_y)

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
pca = decomposition.PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)

for name, label in [('True', 1), ('False', 0)]:
    #ax.text3D(X[y == label, 0].mean(),
    #X[y == label, 1].mean() + 1.5))
    ax.scatter(X[:, 0], X[:, 1], c = np.random.rand(3,), cmap=plt.cm.spectral)
    #ax.w_xaxis.set_ticklabels([])
    #ax.w_yaxis.set_ticklabels([])
    #ax.w_zaxis.set_ticklabels([])

plt.show()

import pylab as pl
pl.scatter(X[:, 0], X[:, 1],  c = np.random.rand(3,))

# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# my_title=('Scatter Plot PCA: %s'%(pca))
# for name, label in [('True', 1), ('False', 0)]:
#     plt.title(my_title +'\n')
# #plt.title('Prediction')
#     plt.scatter(X[,X[1])
#     plt.legend(loc='upper right')
# #plt.xlabel('Actual')
# #plt.ylabel('Predicted')

plt.show()

# Release memory.
plt.clf()
plt.close()


# for name, label in [('True', 1), ('False', 0)]:
#     ax.text3D(X[y == label, 0].mean(),
#               X[y == label, 1].mean() + 1.5,
#               name,
#               horizontalalignment='center',
#               bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# # Reorder the labels to have colors matching the cluster results
# y = np.choose(y, [1, 0]).astype(np.float)
# ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.spectral)
# ax.scatter(X[:, 0], X[:, 1])
# ax.w_xaxis.set_ticklabels([])
# ax.w_yaxis.set_ticklabels([])
# ax.w_zaxis.set_ticklabels([])

#plt.show()


# In[45]:

print(pca.fit(X))


# In[117]:

bbox=dict(alpha=.5, edgecolor='w', facecolor='w')


# In[27]:

X[0],X[1]


# In[243]:

print(pca.fit(X).score)


# In[124]:

print(__doc__)


# Code source: Gaël Varoquaux
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn import decomposition
from sklearn import datasets

np.random.seed(5)

centers = [[1, 1], [-1, -1], [1, -1]]
iris = datasets.load_iris()
X = iris.data
y = iris.target

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)

for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(X[y == label, 0].mean(),
              X[y == label, 1].mean() + 1.5,
              X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.spectral)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()


# In[125]:

X.shape,y.shape


# In[209]:

X_transform.shape[1]


# In[153]:

#sfm.fit(X, y)
n_features = sfm.transform(X).shape[0]
n_features
#df_temp.info()
#sfm.transform(X).shape[1]


# In[ ]:

###Lets Try and see if we have any multicolinearity


# In[87]:

data_X = train_X


# In[153]:

#https://etav.github.io/python/vif_factor_python.html

#Imports
#import pandas as pd
import numpy as np
from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import linear_model
from sklearn import ensemble
from sklearn import metrics



#Steps for Implementing VIF

#Run a multiple regression.
#Calculate the VIF factors.
#Inspect the factors for each predictor variable
#If the VIF is between 5-10, multicolinearity is likely present and you should consider dropping the variable.

#Set up our training and test data

    

def model_predict(data_X,data_y,tag):
    clf = linear_model.LinearRegression()
    clf.fit(data_X, data_y)

    print('\n'*1)

    print('Model')
    
    clf
    
    print('\n'*1)
    
    print('Model Score')

    clf.score(train_X, train_y)

    print('\n'*1)
    
    #Predict Output
    predictions= clf.predict(test_X)

    pred_train = clf.predict(train_X)
    pred_test= clf.predict(test_X)

    print('\n'*1)
        

    print('Coefficients: \n', clf.coef_)
    print('Intercept: \n', clf.intercept_)
    # The mean squared error
    print("Mean squared error: %.2f"
      % np.mean((predictions - test_y) ** 2))

    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % clf.score(test_X, test_y))

    print('\n'*1)
    #Root mean Squared error
    print ('\nRoot Mean Squared Error')

    print np.sqrt(metrics.mean_squared_error(test_y,predictions))


    mae = metrics.mean_absolute_error(test_y, predictions)
    mse = metrics.mean_squared_error(test_y, predictions)

    print('\nMean Absolute Error')
    print(mae)

    print('\nMean Squared Error')
    print(mse)

clf = None

# Create linear regression object
print('Linear Regression all predictors')
tag = 'Linear Model all predictors'

train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
train_y=diamonds_train[['lprice']] # output of our training data
test_X =diamonds_test[['carat','cut','clarity','color','x','y','z','depth','table']] # taking test data features
test_y=diamonds_test[['lprice']]  #output value of test data


model_predict(train_X,train_y,tag)

#gather features
features = "+".join(train_X.columns) # - ["train_y"])
label = "".join(train_y.columns)


#%%capture

# get y and X dataframes based on this regression:
y, X = dmatrices(label + '~' + features, diamonds, return_type='dataframe')

# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns


print(vif.round(1))

print("\n"*3)

clf=None

# Create linear regression object
print('Linear Regression all -depth')
tag = 'Linear Model - depth'


train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','table']]# taking the training data features
train_y=diamonds_train[['lprice']] # output of our training data
test_X =diamonds_test[['carat','cut','clarity','color','x','y','z','table']] # taking test data features
test_y=diamonds_test[['lprice']]  #output value of test data

model_predict(train_X,train_y,tag)


#%%capture
#gather features
features = "+".join(train_X.columns  - ["depth"])

#Pick out labe l                   
label = "".join(train_y.columns)
# get y and X dataframes based on this regression:
y, X = dmatrices(label + '~' + features, diamonds, return_type='dataframe')

# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

print(vif.round(1))

print("\n"*3)

# Create linear regression object
print('Linear Regression all -color')
tag = 'Linear Model - color'

train_X=diamonds_train[['carat','cut','clarity','x','y','z','depth','table']]# taking the training data features
train_y=diamonds_train[['lprice']] # output of our training data
test_X =diamonds_test[['carat','cut','clarity','x','y','z','depth','table']] # taking test data features
test_y=diamonds_test[['lprice']]  #output value of test data

data_y = train_y
model_predict(train_X,train_y,tag)

#clf = linear_model.LinearRegression()


#%%capture
#gather features
features = "+".join(train_X.columns  - ["color"])

#Pick out lable                    
label = "".join(train_y.columns)
# get y and X dataframes based on this regression:
y, X = dmatrices(label + '~' + features, diamonds, return_type='dataframe')

# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

print(vif.round(1))




# Create linear regression object
print('Linear Regression all -minimum')
tag = 'Linear Model - color'

train_X=diamonds_train[['carat','cut','clarity','x','y','z']]# taking the training data features
train_y=diamonds_train[['lprice']] # output of our training data
test_X =diamonds_test[['carat','cut','clarity','x','y','z']] # taking test data features
test_y=diamonds_test[['lprice']]  #output value of test data

data_y = train_y
model_predict(train_X,train_y,tag)

#clf = linear_model.LinearRegression()


#%%capture
#gather features
features = "+".join(train_X.columns  - ["color"])

#Pick out lable                    
label = "".join(train_y.columns)
# get y and X dataframes based on this regression:
y, X = dmatrices(label + '~' + features, diamonds, return_type='dataframe')

# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

print(vif.round(1))



#As expected, there is evidence of multicolinearity.
#A high variance inflation factor indicates they "explain" the same variance within this dataset. 
#We would need to discard one of these variables before moving on to model building or 
#risk building a model with high multicolinearity.


# In[37]:

vif[vif>5]


# In[ ]:




# In[ ]:

# Linear Score
# Linear


# ('Coefficients: \n', array([[-0.43835167,  0.002586  ,  0.02803656, -0.02800461,  0.22426456,
#          0.25589893,  0.16980327,  0.00876309, -0.00236092]]))
# ('Intercept: \n', array([-0.0693151]))
# Mean squared error: 0.01
# Variance score: 0.96



# Root Mean Squared Error
# 0.0901745324925

# Mean Absolute Error
# 0.0679633207308

# Mean Squared Error
# 0.00813144631024
# 16175


# In[ ]:

def ridge_regression(data, predictors, alpha, models_to_plot={}):
    #Fit the model
    ridgereg = Ridge(alpha=alpha,normalize=True)
    ridgereg.fit(data[predictors],data['y'])
    y_pred = ridgereg.predict(data[predictors])
    
    #Check if a plot is to be made for the entered alpha
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(data['x'],y_pred)
        plt.plot(data['x'],data['y'],'.')
        plt.title('Plot for alpha: %.3g'%alpha)
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([ridgereg.intercept_])
    ret.extend(ridgereg.coef_)
    return ret


#Initialize predictors to be set of 15 powers of x
predictors=['x']
predictors.extend(['x_%d'%i for i in range(2,16)])

#Set the different values of alpha to be tested
alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

#Initialize the dataframe for storing coefficients.
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
ind = ['alpha_%.2g'%alpha_ridge[i] for i in range(0,10)]
coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)

models_to_plot = {1e-15:231, 1e-10:232, 1e-4:233, 1e-3:234, 1e-2:235, 5:236}
for i in range(10):
    coef_matrix_ridge.iloc[i,] = ridge_regression(data, predictors, alpha_ridge[i], models_to_plot)



# In[ ]:

#Add the alpha value to the below list if you want to run the algo
a_list = numpy.array([])

for alpha in a_list:
    #Set the base model
    #model = Ridge(alpha=alpha,random_state=seed)
    model = Ridge(alpha=alpha,random_state=23)  
    algo = "Ridge"

    #Accuracy of the model using all features
    for name,i_cols_list in X_all:
        model.fit(train_X[:,i_cols_list],train_y)
        result = mean_absolute_error(numpy.expm1(Y_val), numpy.expm1(model.predict(X_val[:,i_cols_list])))
        mae.append(result)
        print(name + " %s" % result)
        
    comb.append(algo + " %s" % alpha )

#Result obtained by running the algo for alpha=1.0    
if (len(a_list)==0):
    mae.append(1267.5)
    comb.append("Ridge" + " %s" % 1.0 ) 


# In[245]:

test_X,prediction


# In[76]:

#import numpy as np
import statsmodels.api as sm
import pylab

#test = np.random.normal(0,1, 1000)

sm.qqplot(pred_test, line='45')
pylab.show()




# In[86]:

linear.coef_
pd.DataFrame(test_X)

#0 	-0.43799 	0.002602 	0.027973 	-0.027501 	0.221848 	0.234685 	0.206788 	0.006702 	-0.002407


# In[85]:

pd.DataFrame(linear.coef_)


# In[87]:

test_X.columns


# In[78]:

pd.DataFrame(zip(test_X.columns,linear.coef_),columns=['predictors','estimatesCoefficient'])


# In[79]:

pd.DataFrame(zip(test_X.columns,(linear.coef_)),columns=['predictors','estimatesCoefficient'])


# In[81]:

pd.DataFrame(zip(test_y,predictions),columns=['lprice','predicted value'])

#np.exp(3.617734)

#pred_df['Price']=np.exp(test_y)

#pred_df


# In[82]:

from math import exp, expm1, log
pd.DataFrame(zip(test_y,predictions),columns=['lprice','predicted value'])

#log(10,3.617734)


# # Our Classification Models on Ideal cut (True/False)

# In[87]:

# importing alll the necessary packages to use the various classification algorithms
from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm
from sklearn.cross_validation import train_test_split #to split the dataset for training and testing
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm

seed=(1976)


#We have 'cut' included but the target is ideal_flag...a predictor which define the ideal_flag
#We have dropped 'cut' as the target is ideal_flag which is based on cut value
train_X = diamonds_train[['carat','clarity','color','x','y','z','depth','table','lprice']]# taking the training data features
train_y = diamonds_train.ideal_flag# output of our training data
test_X = diamonds_test[['carat','clarity','color','x','y','z','depth','table','lprice']] # taking test data features
test_y = diamonds_test.ideal_flag   #output value of test data

#We have 'cut' included but the target is ideal_flag...a predictor which define the ideal_flag
#We have dropped 'cut' as the target is ideal_flag which is based on cut value
classification_price_train_X = diamonds_train[['carat','clarity','color','x','y','z','depth','table','price']]# taking the training data features
classification_price_train_y = diamonds_train.ideal_flag# output of our training data
classification_price_test_X = diamonds_test[['carat','clarity','color','x','y','z','depth','table','price']] # taking test data features
classification_price_test_y = diamonds_test.ideal_flag   #output value of test data


# In[94]:

#based on http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

#Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.


import itertools
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics


class_names = ['True','False']

model = svm.SVC() #select the algorithm
model.fit(classification_price_train_X,classification_price_train_y) # we train the algorithm with the training data and the training output
prediction = model.predict(classification_price_test_X) #now we pass the testing data to the trained algorithm

print('Classification Models using price')
print('The accuracy of the SVM is:',metrics.accuracy_score(prediction,classification_price_test_y))#now we check the accuracy of the algorithm. 
#we pass the predicted output by the model and the actual output


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(classification_price_test_y, prediction)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

# Release memory.
plt.clf()
plt.close()


# In[69]:

#based on http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

#Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.


import itertools
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics


class_names = ['True','False']

model = svm.SVC() #select the algorithm
model.fit(train_X,train_y) # we train the algorithm with the training data and the training output
prediction = model.predict(test_X) #now we pass the testing data to the trained algorithm

print('Classification Models using lprice')
print('The accuracy of the SVM is:',metrics.accuracy_score(prediction,test_y))#now we check the accuracy of the algorithm. 
#we pass the predicted output by the model and the actual output


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(test_y, prediction)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

# Release memory.
plt.clf()
plt.close()



# # Combined Classification models using price

# In[95]:

#ased on on https://www.kaggle.com/ash316/ml-from-scratch-with-iris

import itertools
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def model_run(model_name):
    model = model_name#svm.SVC() #select the algorithm
    model.fit(classification_price_train_X,classification_price_train_y) # we train the algorithm with the training data and the training output
    prediction=model.predict(classification_price_test_X) #now we pass the testing data to the trained algorithm
    print("\n"*3)
    print('Classification Models using price')
    print model_name
    print("\n"*3)
    if str(model).startswith('SVC'):
        print('The accuracy of the SVM is:',metrics.accuracy_score(prediction,test_y))#now we check the accuracy of the algorithm. 
        print("\n"*3)
    if str(model).startswith('LogisticRegression'):
        print('The accuracy of the Logistical Regression is:',metrics.accuracy_score(prediction,test_y))#now we check the accuracy of the algorithm. 
        print("\n"*3)
    if str(model).startswith('DecisionTreeClassifier'):
        print('The accuracy of the Decision Tree is:',metrics.accuracy_score(prediction,test_y))#now we check the accuracy of the algorithm. 
        print("\n"*3)
    if str(model).startswith('KNeighborsClassifier'):
        print('The accuracy of the K nearesr Neighbour is:',metrics.accuracy_score(prediction,test_y))#now we check the accuracy of the algorithm. 
        print("\n"*3)
    
# if model_name=='KNeighborsClassifier(n_neighbors=2)':   
    
    #we pass the predicted output by the model and the actual output

    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(classification_price_test_y, prediction)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix

    plt.figure()
    #plt.subplot(2,1,1)
    #plt.figsize=(6, 3)
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    
    plt.figure()
    #plt.subplot(2,1,2)
    #plt.figsize=(6, 3)
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
    plt.show()
 
    
class_names = ['True','False']


#Run the model and build the confusion matrix for each

model_run(svm.SVC())
model_run(LogisticRegression())
model_run(DecisionTreeClassifier())
model_run(KNeighborsClassifier(n_neighbors=2))


# # Combined Classification models using lprice

# In[83]:

#ased on on https://www.kaggle.com/ash316/ml-from-scratch-with-iris

import itertools
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def model_run(model_name):
    model = model_name#svm.SVC() #select the algorithm
    model.fit(train_X,train_y) # we train the algorithm with the training data and the training output
    prediction=model.predict(test_X) #now we pass the testing data to the trained algorithm
    print("\n"*3)
    print('Classification Models using lprice')
    print model_name
    print("\n"*3)
    if str(model).startswith('SVC'):
        print('The accuracy of the SVM is:',metrics.accuracy_score(prediction,test_y))#now we check the accuracy of the algorithm. 
        print("\n"*3)
    if str(model).startswith('LogisticRegression'):
        print('The accuracy of the Logistical Regression is:',metrics.accuracy_score(prediction,test_y))#now we check the accuracy of the algorithm. 
        print("\n"*3)
    if str(model).startswith('DecisionTreeClassifier'):
        print('The accuracy of the Decision Tree is:',metrics.accuracy_score(prediction,test_y))#now we check the accuracy of the algorithm. 
        print("\n"*3)
    if str(model).startswith('KNeighborsClassifier'):
        print('The accuracy of the K nearesr Neighbour is:',metrics.accuracy_score(prediction,test_y))#now we check the accuracy of the algorithm. 
        print("\n"*3)
    
# if model_name=='KNeighborsClassifier(n_neighbors=2)':   
    
    #we pass the predicted output by the model and the actual output

    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(test_y, prediction)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix

    plt.figure()
    #plt.subplot(2,1,1)
    #plt.figsize=(6, 3)
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    
    plt.figure()
    #plt.subplot(2,1,2)
    #plt.figsize=(6, 3)
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
    plt.show()
 
    
class_names = ['True','False']


#Run the model and build the confusion matrix for each

model_run(svm.SVC())
model_run(LogisticRegression())
model_run(DecisionTreeClassifier())
model_run(KNeighborsClassifier(n_neighbors=2))


# In[88]:


from sklearn import svm, datasets
from sklearn import metrics
model = svm.SVC() #select the algorithm
model.fit(train_X,train_y) # we train the algorithm with the training data and the training output
prediction=model.predict(test_X) #now we pass the testing data to the trained algorithm
print('The accuracy of the SVM is:',metrics.accuracy_score(prediction,test_y))#now we check the accuracy of the algorithm. 
#we pass the predicted output by the model and the actual output


# In[ ]:

#SVM is giving very good accuracy . We will continue to check the accuracy for different models.

#Now we will follow the same steps as above for training various machine learning algorithms.



# In[93]:

from sklearn import svm, datasets
from sklearn import metrics

#SVM 
model = svm.SVC() #select the algorithm
model.fit(train_X,train_y) # we train the algorithm with the training data and the training output
prediction=model.predict(test_X) #now we pass the testing data to the trained algorithm

print('The accuracy of the SVM is:',metrics.accuracy_score(prediction,test_y))#now we check the accuracy of the algorithm. 
#we pass the predicted output by the model and the actual output

#SVM is giving very good accuracy . We will continue to check the accuracy for different models.

#Now we will follow the same steps as above for training various machine learning algorithms.


print('\n'*1)

#Logistic Regression
model = LogisticRegression()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,test_y))

print('\n'*1)

#Decision Tree

model=DecisionTreeClassifier()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction,test_y))

print('\n'*1)

#K-Nearest Neighbours
model=KNeighborsClassifier(n_neighbors=2) #n_neighbours=2 means we are trying to split them into 2 clusters
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('The accuracy of the KNN is',metrics.accuracy_score(prediction,test_y))


# In[ ]:




# In[65]:

import sklearn; print(sklearn.__file__)
(sklearn.__version__) 


# In[80]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
#from sklearn.model_selection import GridSearchCV

from sklearn import grid_search #GridSearchCV


#We have 'cut' included but the target is ideal_flag...a predictor which define the ideal_flag
#We have dropped 'cut' as the target is ideal_flag which is based on cut value
train_X = diamonds_train[['carat','clarity','color','x','y','z','depth','table','lprice']]# taking the training data features
train_y = diamonds_train.ideal_flag# output of our training data
test_X = diamonds_test[['carat','clarity','color','x','y','z','depth','table','lprice']] # taking test data features
test_y = diamonds_test.ideal_flag   #output value of test data




# In[82]:

# Choose the type of classifier. 
clf = RandomForestClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = grid_search.GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(train_X, train_y)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
clf.fit(train_X, train_y)


# In[14]:

#accuracy of our randomForest

predictions = clf.predict(test_X)
print(accuracy_score(test_y, predictions))

print('The accuracy of the RandomForestClassifier is',metrics.accuracy_score(prediction,test_y))



# In[29]:

print(diamonds_train.shape)
print(diamonds_test.shape)


# In[30]:

from sklearn.cross_validation import KFold

#train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]# taking the training data features
#train_Y=train.Species# output of our training data
#test_X = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] # taking test data features
#test_Y =test.Species   #output value of test data

#X_all = diamonds.drop(['ideal_flag'], axis=1)
#Y_all = diamonds['ideal_flag']

X_all = train_X
Y_all = train_y



def run_kfold(clf):
    kf = KFold(150, n_folds=10)
    outcomes = []
    fold = 0
    for train_index, test_index in kf:
        fold += 1
        train_X, test_X = X_all.values[train_index], X_all.values[test_index]
        train_y, test_y = Y_all.values[train_index], Y_all.values[test_index]
        clf.fit(train_X, train_y)
        predictions = clf.predict(test_X)
        accuracy = accuracy_score(test_y, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))     
    mean_outcome = np.mean(outcomes)
    print("Mean Accuracy: {0}".format(mean_outcome)) 

run_kfold(clf)


# In[31]:

#Predict the Actual Test Data

#And now for the moment of truth. Make the predictions, export the CSV file, and upload them to Kaggle.
print('KFold using lprice')
prediction = clf.predict(test_X)
print('The accuracy of the last KFold is',metrics.accuracy_score(prediction,test_y))

print('\n'*1)
    
# print('KFold using price')
# prediction = clf.predict(test_X)
# print('The accuracy of the last KFold is',metrics.accuracy_score(prediction,test_y))




# In[64]:

#We have 'cut' included but the target is ideal_flag...a predictor which define the ideal_flag
#We have dropped 'cut' as the target is ideal_flag which is based on cut value
train_X=diamonds_train[['carat','clarity','color','x','y','z','depth','table','lprice']]# taking the training data features
train_y=diamonds_train.ideal_flag# output of our training data
test_X=diamonds_test[['carat','clarity','color','x','y','z','depth','table','lprice']] # taking test data features
test_y=diamonds_test.ideal_flag   #output value of test data



#Import Library

#Import other necessary libraries like pandas, numpy...
from sklearn import tree
from sklearn import metrics
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create tree object 
#model = tree.DecisionTreeClassifier(criterion='gini') # for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini  
model = tree.DecisionTreeRegressor() #for regression
# Train the model using the training sets and check score
model.fit(train_X, train_y)
model.score(train_X, train_y)
#Predict Output
predicted= model.predict(test_X)

print('The accuracy of the DecisionTreeRegressor is',metrics.accuracy_score(prediction,test_y))



# In[ ]:

Clustering


# In[14]:


#from sklearn.cluster import KMeans
#kcluster = KMeans(n_clusters = 6)

# apply the kmeans clustering
#kcluster(diamonds)


from sklearn.cluster import KMeans
import numpy as np

#X = np.array(train_X)
#y = np.array(train_y)

X_train = np.array(train_X)
y_train = np.array(train_y)

X_test = np.array(test_X)
y_test = np.array(test_y)



kmeans = KMeans(n_clusters=6, random_state=0).fit(X_train)
kmeans.labels_

kmeans.predict(X_test)

kmeans.cluster_centers_




# In[15]:


X_train = np.array(train_X)
y_train = np.array(train_y)

X_test = np.array(test_X)
y_test = np.array(test_y)


range_n_clusters = [2, 3, 4, 5, 6]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X_train) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X_train)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X_train, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X_train, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values =             sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors)

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1],
                marker='o', c="white", alpha=1, s=200)

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.show()


# # Rough Code to see where my error is in Lasso

# In[ ]:




# In[32]:

#Linear regression
#Import Library
#Import other necessary libraries like pandas, numpy
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn import metrics


def model_predict(clf,tag):
    
    if tag == 'linear':
        
        print('\n'*1)
        print ('#################### Linear Model #################### ')
        
    if tag == 'ridge':
        clf.alpha = 0.0
        print('\n'*1)
        print ('/n#################### Ridge Model #################### ')
        
    if tag == 'lasso':

        #from sklearn import linear_model, datasets
        #clf = linear_model.LassoCV()

        #clf.fit(train_X, train_y)
        #LassoCV(alphas=array([ 2.14804,  2.00327, ...,  0.0023 ,  0.00215]),
        #LassoCV(alphas=array([ 2.14804,  2.00327, ...,  0.0023 ,  0.00215]),
        #    copy_X=True, cv=None, eps=0.001, fit_intercept=True, max_iter=1000,
        #    n_alphas=100, normalize=False, precompute='auto', tol=0.0001,
        #    verbose=False)
        # The estimator chose automatically its lambda:
        #clf.alpha  
        
        #we set as 1.0
        #clf.alpha = 1.0
        #clf.fit(train_X, train_y)
    
    
          
        print('\n'*1)
        print ('/n#################### Lasso Model #################### ')

    
    # Train the model using the training sets and check score
    clf.fit(train_X, train_y)

    print('\n'*1)

    print('Model')
    
    print(clf)
    
    print('\n'*1)

    print('Model Score')

    clf.score(train_X, train_y)

    #Predict Output
    predictions= clf.predict(test_X)

    pred_train = clf.predict(train_X)
    pred_test= clf.predict(test_X)

    
    print('\n'*1)

    print('Coefficients: \n', clf.coef_)
    print('Intercept: \n', clf.intercept_)
    # The mean squared error
    print("Mean squared error: %.2f"
      % np.mean((predictions - test_y) ** 2))

    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % clf.score(test_X, test_y))

    print('\n'*1)
    #Root mean Squared error
    print ('\nRoot Mean Squared Error')

    print np.sqrt(metrics.mean_squared_error(test_y,predictions))


    mae = metrics.mean_absolute_error(test_y, predictions)
    mse = metrics.mean_squared_error(test_y, predictions)

    print('\nMean Absolute Error')
    print(mae)

    print('\nMean Squared Error')
    print(mse)

    pred_len=int(len(predictions))
    print(pred_len)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Prediction')

    plt.scatter(test_y, pred_test, 
         label='Prediction Accuracy')
    plt.legend(loc='upper right')
    #plt.xlabel('Actual')
    #plt.ylabel('Predicted')

    plt.show()
    
        
    #Result obtained after running the algo. Comment the below two lines if you want to run the algo
    mae_list.append(mae)
    comb.append(tag)  

   
    

#List of combinations
comb = []

#Dictionary to store the MAE for all algorithms 
mae_list = []

clf=None

# Create linear regression object
print('Linear Score')
tag = 'linear'

train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
train_y=diamonds_train[['lprice']] # output of our training data
test_X =diamonds_test[['carat','cut','clarity','color','x','y','z','depth','table']] # taking test data features
test_y=diamonds_test[['lprice']]  #output value of test data


clf = linear_model.LinearRegression()
#create models and train
model_predict(clf,tag)



clf=None
# Create Ridge regression model and train
print('Ridge Score')
tag = 'ridge'

train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
train_y=diamonds_train[['lprice']] # output of our training data
test_X =diamonds_test[['carat','cut','clarity','color','x','y','z','depth','table']] # taking test data features
test_y=diamonds_test[['lprice']]  #output value of test data



clf = linear_model.Ridge(alpha = 1.0)
#create models and train
model_predict(clf,tag)


#Plot the MAE of all combinations
fig, ax = plt.subplots()
plt.plot(mae_list)
#Set the tick names to names of combinations
ax.set_xticks(range(len(comb)))
ax.set_xticklabels(comb,rotation='vertical')
#Plot the accuracy for all combinations
plt.show()    

# Release memory.
plt.clf()
plt.close()


##### not working 
clf=None
#create Lasso model and train
print('Lasso Score')
tag = 'lasso'

train_X=diamonds_train[['carat','cut','clarity','color','x','y','z','depth','table']]# taking the training data features
train_y=diamonds_train[['lprice']] # output of our training data
test_X =diamonds_test[['carat','cut','clarity','color','x','y','z','depth','table']] # taking test data features
test_y=diamonds_test[['lprice']]  #output value of test data




clf=None

alpha = 1.0
#alpha=1.0

#clf = Lasso(alpha = alpha)
#clf = Lasso(alpha=alpha)

print('Lasso Model')

print(clf)


###############################

alpha = 0.1
lasso = Lasso(alpha=alpha)
clf=lasso
#pred_test = lasso.fit(train_X, train_y).predict(test_X)

#print(pred_test)

#create models and train
model_predict(clf,tag)


#Plot the MAE of all combinations
fig, ax = plt.subplots()
plt.plot(mae_list)
#Set the tick names to names of combinations
ax.set_xticks(range(len(comb)))
ax.set_xticklabels(comb,rotation='vertical')
#Plot the accuracy for all combinations
plt.show()    

# Release memory.
plt.clf()
plt.close()


# In[16]:



#Linear regression
#Import Library
#Import other necessary libraries like pandas, numpy
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn import metrics


def show_coefficients(model):

    if model == 'linear':
        print ('Linear')
         
        # #Equation coefficient and Intercept
        #The coefficients

        print('\n'*1)

        print('Coefficients: \n', linear.coef_)
        print('Intercept: \n', linear.intercept_)
        # The mean squared error
        print("Mean squared error: %.2f"
          % np.mean((predictions - test_y) ** 2))

        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % linear.score(test_X, test_y))

        print('\n'*1)
        #Root mean Squared error
        print ('\nRoot Mean Squared Error')

        print np.sqrt(metrics.mean_squared_error(test_y,predictions))


        mae = metrics.mean_absolute_error(test_y, predictions)
        mse = metrics.mean_squared_error(test_y, predictions)

        print('\nMean Absolute Error')
        print(mae)

        print('\nMean Squared Error')
        print(mse)

        pred_len=int(len(predictions))
        print(pred_len)

        #Result obtained after running the algo. Comment the below two lines if you want to run the algo
        mae_list.append(mae)
        comb.append("LR" )  
        
    else:
        print ('Not')
        # The coefficients
        print('\n'*1)

        print('Coefficients: \n', clf.coef_)
        print('Intercept: \n', clf.intercept_)
        # The mean squared error
        print("Mean squared error: %.2f"
              % np.mean((predictions - test_y) ** 2))

        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % clf.score(test_X, test_y))

        print('\n'*1)
        #Root mean Squared error
        print ('\nRoot Mean Squared Error')

        print np.sqrt(metrics.mean_squared_error(test_y,predictions))

        #print('###############')

        #model = LinearRegression().fit(X_train, y_train)
        #predictions = model.predict(X_test)
        mae = metrics.mean_absolute_error(test_y, predictions)
        mse = metrics.mean_squared_error(test_y, predictions)

        print('\nMean Absolute Error')
        print(mae)

        print('\nMean Squared Error')
        print(mse)

        pred_len=int(len(predictions))
        print(pred_len)

        #Result obtained after running the algo. Comment the below two lines if you want to run the algo
        mae_list.append(mae)
        comb.append("Ridge" )   




#List of combinations
comb = []

#Dictionary to store the MAE for all algorithms 
mae_list = []


# Create linear regression object
model = 'linear'
linear = linear_model.LinearRegression()
# Train the model using the training sets and check score
linear.fit(train_X, train_y)

print('\n'*1)

print('Linear Score')

linear.score(train_X, train_y)


#Predict Output
predictions= linear.predict(test_X)

pred_train = linear.predict(train_X)
pred_test= linear.predict(test_X)

show_coefficients('linear')



# Create Ridge regression model and train
print('Ridge Score')
model = 'ridge'
#create models and train
clf = Ridge(alpha = 1.0)
#Ridge
clf.fit(train_X, train_y)#Ridge

#evaluate on development set
#Predict Output
predictions= clf.predict(test_X)

show_coefficients('ridge')
#Y = clf.predict(X_dev)

 
#sq_diff = np.square(np.log(prices_dev) - np.log(Y))
#error = np.sqrt(np.sum(sq_diff) / prices_dev.shape[0])
#error


#create Lasso model and train
model = 'lasso'
print('Lasso Score')
clf = None
clf = Lasso(alpha = 1.0)
clf
clf.fit(train_X, train_y)

#evaluate on development set
#Predict Output
predictions= clf.predict(test_X)
#Y = clf.predict(X_dev)

# The coefficients
print('\n'*1)

print('Coefficients: \n', clf.coef_)
print('Intercept: \n', linear.intercept_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((predictions - test_y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % clf.score(test_X, test_y))

print('\n'*1)
#Root mean Squared error
print ('\nRoot Mean Squared Error')

print np.sqrt(metrics.mean_squared_error(test_y,predictions))

#print('###############')

#model = LinearRegression().fit(X_train, y_train)
#predictions = model.predict(X_test)
mae = metrics.mean_absolute_error(test_y, predictions)
mse = metrics.mean_squared_error(test_y, predictions)

print('\nMean Absolute Error')
print(mae)

print('\nMean Squared Error')
print(mse)

pred_len=int(len(predictions))
print(pred_len)

#Result obtained after running the algo. Comment the below two lines if you want to run the algo
mae_list.append(mae)
comb.append("Lasso" )    


#Plot the MAE of all combinations
fig, ax = plt.subplots()
plt.plot(mae_list)
#Set the tick names to names of combinations
ax.set_xticks(range(len(comb)))
ax.set_xticklabels(comb,rotation='vertical')
#Plot the accuracy for all combinations
plt.show()    

# Release memory.
plt.clf()
plt.close()


#pred_len=(len(predictions))


#fig, ax = plt.subplots()
#for a in [predictions, test_y]:
    #sns.distplot(a, bins=range(len(predictions)), ax=ax, kde=True)
#ax.set_xlim([0, 100])


#for col_id in predictions.columns:
    #sns.distplot(predictions[col_id])

#plt.hist(predictions, alpha=0.5, color='red', cumulative=True, normed=True, bins=16175, histtype='stepfilled', stacked=True)

#plt.show()

#plt.hist(test_y, alpha=0.5, color='blue', cumulative=True, normed=True, bins=16175, histtype='stepfilled', stacked=True)
#plt.show()
#_ = plt.hist(predictions, alpha=0.5, color='red', cumulative=True, normed=True, bins=len(predictions), histtype='stepfilled', stacked=True)
#_ = plt.hist(test_y, alpha=0.5, color='blue', cumulative=True, normed=True, bins=len(predictions), histtype='stepfilled', stacked=True)



#caret
#AIC
#AIC(linear)

#linear.fit(X_parameters, Y_parameters)
#plt.scatter(train_X, train_y,color='blue')
#plt.plot(train_X,linear.predict(test_X),color='red',linewidth=4)
#plt.xticks(())
#plt.yticks(())
#plt.show()

# Plot outputs
#plt.scatter(test_X, test_y,  color='black')
#plt.plot(test_X, prediction, color='blue',linewidth=3)

#plt.xticks(())
#plt.yticks(())

#plt.show()


#accuracy of our Linear Regression  Can't use metrics.accuracy_score
#print('The accuracy of the Linear Regression is',metrics.accuracy_score(prediction,test_y))

#Can't use continuous and binary  so clf is out
#predictions = clf.predict(test_X)
#print(accuracy_score(test_y, predictions))

#print classification_report(test_y, prediction)

#from sklearn.metrics import classification_report
#classificationReport = classification_report(test_y, prediction)# target_names=target_names)

#plot_classification_report(classificationReport)



#Linear Score
#('Coefficient: \n', array([[-0.43835167,  0.002586  ,  0.02803656, -0.02800461,  0.22426456,
         #0.25589893,  0.16980327,  0.00876309, -0.00236092]]))
#('Intercept: \n', array([-0.0693151]))


#('Coefficients: \n', array([[-0.43835167,  0.002586  ,  0.02803656, -0.02800461,  0.22426456,
         #0.25589893,  0.16980327,  0.00876309, -0.00236092]]))
#Mean squared error: 0.01
#Variance score: 0.96



#Root Mean Squared Error
#0.0901745324925

#Mean Absolute Error
#0.0679633207308

#Mean Squared Error
#0.00813144631024
#16175


# In[ ]:

# Citations
# Seabold, Skipper, and Josef Perktold. “Statsmodels: Econometric and statistical modeling with python.” Proceedings of the 9th Python in Science Conference. 2010.

