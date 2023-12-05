#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd                                 ### importing libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os        ### change to working directory


# In[10]:


data = pd.read_csv("C:/Users/Yashwanth G R/Downloads/Market-Segmentation-of-EVs-in-India-main/Market-Segmentation-of-EVs-in-India-main/EV data/ElectricCarData_Norm.csv")         ### load the data


# In[11]:


data.head()


# In[12]:


Data_N=data.loc[:,['Brand','Accel','TopSpeed','Range','Efficiency','FastCharge','RapidCharge','PowerTrain','PlugType','BodyStyle','Segment']]
E=[]
for c in Data_N.columns:
       dic={}
       F=[]
       for j in range(len(Data_N[c].unique())):         ### Converting catgorical data into numeric data
            dic[Data_N[c].unique()[j]]=j    
       for i in range(len(Data_N[c])):
                F.append(dic[Data_N[c][i]])
       E.append(F)


# In[13]:


E=np.array(E).reshape((103,11))


# In[14]:


Num_data = pd.DataFrame(E,columns=Data_N.columns)


# In[15]:


Num_data[['Seats','PriceEuro']]=np.array(data[['Seats','PriceEuro']])


# In[16]:


Num_data        ### numeric data


# In[17]:


sns.pairplot(Num_data)     ### scatter plot against every variable


# In[18]:


plt.figure(figsize=(10,8))           ### check for correlation
sns.heatmap(Num_data.corr(),annot=True)
plt.show()


# ### Descriptive Analysis

# In[19]:


plt.scatter(Num_data['Range'],Num_data['PriceEuro'])
plt.ylabel('Price')
plt.xlabel('Range')


# In[25]:


# Standardizing the data
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
P = sc.fit_transform(np.array(Num_data))
S = sc.fit_transform(np.array(data['Seats']).reshape(-1,1))


# In[26]:


P = pd.DataFrame(P,columns=Num_data.columns)
P


# In[27]:


plt.scatter(P['Range'],P['PriceEuro'])


# In[28]:


#x = sc.fit_transform(Num_data)
#X = pd.DataFrame(x,columns=Num_data.columns)
sns.pairplot(P)


# In[29]:


plt.scatter(S,P)
plt.ylabel('Price')
plt.xlabel('Number of seats')


# In[30]:


data.isnull().sum()      ### check the missing values


# ### Types of Car Brands

# In[31]:


Brands = data['Brand'].unique()
L=data['Brand'].value_counts()
plt.figure(figsize=(15,8))
plt.pie(L,labels=Brands)                                        ### Pie chart
plt.legend(Brands,title='cars',bbox_to_anchor =(1, 0, 0.5, 1))
plt.show()


# In[32]:


Brands


# ### Cars with rapid charge

# In[33]:


Data_brand = Brands[:]          ### Barplots
rp=[]
No_rp=[]
for brand in Data_brand:
    D = data[data['Brand']==brand]
    l=len(D['RapidCharge'][D['RapidCharge']=='Rapid charging possible'])
    n = len(D['RapidCharge'][D['RapidCharge']=='Rapid charging not possible'])
    rp.append(l)
    No_rp.append(n)
    
X_axis = np.arange(len(Data_brand))
plt.figure(figsize=(30,10))
plt.bar(X_axis - 0.2, rp, 0.4, label = 'Rapidcharge')
plt.bar(X_axis + 0.2, No_rp, 0.4, label = 'No rapidcharge')
plt.xticks(X_axis, Data_brand)
plt.xlabel("Car Brands")
plt.ylabel("Number of cars")
plt.title("Cars with rapid charge")
plt.legend()
plt.show()


# ### price segments

# In[34]:


Segments=[]
for segment in data['Segment'].unique():
    Segments.append(len(data[data['Segment']==segment]))
plt.bar(data['Segment'].unique(),Segments)
plt.title('Price Segments')
plt.xlabel('Segments')
plt.ylabel('Number of cars')
plt.show()


# ### Price range for respective segments 

# In[35]:


Avg_price=[]
for segment in data['Segment'].unique():
    Avg_price.append(data[data['Segment']==segment]['PriceEuro'].mean())
plt.bar(data['Segment'].unique(),Avg_price)
plt.title('Price Segments')
plt.xlabel('Segments')
plt.ylabel('Avg_price')
plt.show()


# ### Price Segments with respect to each brand

# In[36]:


for brand in Data_brand:
    Segments=[]
    for segment in data['Segment'].unique():
        D=data[data['Brand']==brand]
        l = len(D['Segment'][D['Segment']==segment])
        Segments.append(l)
    plt.bar(data['Segment'].unique(),Segments)
    plt.title(brand)
    plt.xlabel('Segments')
    plt.ylabel('Number of cars')
    plt.show()


# ### Capacity with respect to price segments

# In[37]:


for segment in data['Segment'].unique():
    D=data[data['Segment']==segment]
    Capacity=[]
    for seats in data['Seats'].unique():
       Capacity.append(len(D['Seats'][D['Seats']==seats]))
    plt.bar(data['Seats'].unique(),Capacity)
    plt.title(segment)
    plt.xlabel('Number of seats')
    plt.ylabel('Number of cars')
    plt.show()


# ### Body type with respect to number of seats and segments

# In[38]:


for car_type in data['BodyStyle'].unique():
    D=data[data['BodyStyle']==car_type]
    Seats=[]
    Car_type=[]
    for seats in data['Seats'].unique():
       Seats.append(len(D['Seats'][D['Seats']==seats]))
    for segment in data['Segment'].unique():
       Car_type.append(len(D['Segment'][D['Segment']==segment]))
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.bar(data['Seats'].unique(),Seats)
    plt.title(car_type)
    plt.xlabel('Number of seats')
    plt.ylabel('Number of cars')
    plt.subplot(1,2,2)
    plt.bar(data['Segment'].unique(),Car_type)
    plt.title(car_type)
    plt.xlabel('Segments')
    plt.ylabel('Number of cars')    
    plt.show()


# ### PCA 

# In[39]:


from sklearn.decomposition import PCA   ### import PCA to perform pricipal component analysis
pca = PCA()
df = pca.fit_transform(Num_data)
explained_variance = pca.explained_variance_ratio_    ### get the variance associated with each and every pca variable


# In[40]:


explained_variance         ### Variance of the varibles after PCA transformation


# In[41]:


plt.figure(figsize=(10,8))                ### plot to see the variations of the transformed variables
plt.plot(np.cumsum(explained_variance))
plt.grid()
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance')
sns.despine()


# In[42]:


df = pd.DataFrame(df)   ## Transformed variables after performing PCA
df


# In[43]:


from numpy.linalg import norm                ### cosine similarity to check if there is any relation between variables
def cosine_similarity(A,B):
    x = norm(A, axis=1).reshape(-1,1)
    cosine = np.dot(A,B)/(x*norm(B))
    return cosine


# In[44]:


Data = Num_data.astype('float64')


# In[45]:


similarity_matrix = cosine_similarity(Data,np.transpose(Data))


# In[46]:


similarity_matrix


# In[47]:


def euclidian_distance(A):         ### euclidean distance to check if there is any relation between variables
    L1=[]
    A = np.array(A)
    for i in range(len(A)):
        L=[]
        for j in range(len(A)):
            x=A[i,:]-A[j,:]
            x = np.dot(np.transpose(x),x)
            L.append(x)
        L1.append(L)
    return np.array(L1)


# In[48]:


Dis = euclidian_distance(Data)


# In[49]:


Dis      ### euclidean matrix


# In[50]:


plt.figure(figsize=(10,8))
sns.heatmap(Dis,annot=False)


# ### Geographic Aspects 

# In[52]:


EV_count_data = pd.read_csv("C:/Users/Yashwanth G R/Downloads/Market-Segmentation-of-EVs-in-India-main/Market-Segmentation-of-EVs-in-India-main/EV data/EV_count.csv")
EV_count_data.head()


# In[53]:


plt.figure(figsize=(20,8))
DF = EV_count_data[['States/Uts','Total vehicle count']].sort_values('Total vehicle count', ascending=False)
plt.bar(DF['States/Uts'][1:15],DF['Total vehicle count'][1:15])
plt.xlabel('States')
plt.ylabel('Vehicle count')
plt.show()


# In[54]:


Type_wheeler = EV_count_data.iloc[:-1,2:7]
Type_wheeler = Type_wheeler.replace('-',0).astype('int64')
Sum_type = Type_wheeler.sum(axis=0,skipna=True)
plt.figure(figsize=(20,8))
plt.bar(Type_wheeler.columns,Sum_type)
plt.xlabel('vehicle_type')
plt.ylabel('Number of vehicles')
plt.show()


# In[55]:


L=Sum_type
plt.figure(figsize=(10,8))
plt.pie(L,labels=Type_wheeler.columns)
plt.legend(Type_wheeler.columns,title='Vehicle types',bbox_to_anchor =(1, 0, 0.5, 1))
plt.show()


# In[62]:


Geographic_data = pd.read_csv("C:/Users/Yashwanth G R/Downloads/Market-Segmentation-of-EVs-in-India-main/Market-Segmentation-of-EVs-in-India-main/EV data/electric_vehicle_charging_station_list.csv")


# In[63]:


Geographic_data.head()


# In[64]:


Places = Geographic_data['region'].unique()
L=Geographic_data['region'].value_counts()
plt.figure(figsize=(15,8))
plt.pie(L,labels=Places)
plt.legend(Places,title='places',bbox_to_anchor =(1, 0, 0.5, 1))
plt.show()


# In[65]:


L1=[]
L2=[]
L3=[]
for place in Geographic_data['region'].unique():
    D = Geographic_data[Geographic_data['region']==place]
    l1=len(D['power'][D['power']=='15 kW'])
    l2 = len(D['power'][D['power']=='142kW'])
    l3 = len(D['power'][D['power']=='10(3.3 kW each)'])
    L1.append(l1)
    L2.append(l2)
    L3.append(l3)
    
X_axis = np.arange(len(Places))
plt.figure(figsize=(15,6))
plt.bar(X_axis - 0.1,L1, 0.2, label = '15 kW')
plt.bar(X_axis + 0.1,L2, 0.2, label = '142kW')
plt.bar(X_axis + 0.2,L3, 0.2, label = '10(3.3 kW each)')
plt.xticks(X_axis, Places)
plt.xlabel("Location")
plt.ylabel("Number of places")
plt.title("Power Stations")
plt.legend()
plt.show()


# ### K means

# In[66]:


from sklearn.cluster import KMeans


# In[67]:


Num_data['Brand'].max()


# In[68]:


kmeans = KMeans(n_clusters=6, random_state=0).fit(P.loc[:,['Segment','Seats']])   ### assume inital number of clusters
identified_clusters = kmeans.predict(P.iloc[:,3:5])


# In[69]:


data_with_clusters = pd.DataFrame(np.array(P.loc[:,['Segment','Seats']]),columns=['Segment','Seats'])    ### plot the clusters 
data_with_clusters['Clusters'] = identified_clusters 
#plt.figure(figsize=(10,8))
plt.scatter(data_with_clusters['Segment'],data_with_clusters['Seats'],c=data_with_clusters['Clusters'],cmap='rainbow')
plt.xlabel('Segment')
plt.ylabel('Seats')
plt.show()


# In[70]:


wcss=[]                       ### using elbow method to get optimal cluster nummber
for i in range(1,5):
 kmeans = KMeans(i)
 kmeans.fit(Num_data.iloc[:,3:5])
 wcss_iter = kmeans.inertia_
 wcss.append(wcss_iter)
 
number_clusters = range(1,5)
plt.plot(number_clusters,wcss)
plt.title('The Elbow title')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[61]:


Num_data.loc[:,['Range','PriceEuro']]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




