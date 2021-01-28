#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt


# In[3]:


movie=pd.read_csv("my_movies.csv")
movie.head()


# In[20]:


dummy=pd.get_dummies(movie[["V1","V2","V3","V4","V5"]])
dummy.head()


# ### Apriori Algorithm

# In[8]:


frequent_itemsets=apriori(dummy,min_support=0.1,use_colnames=True)
frequent_itemsets


# In[9]:


rules=association_rules(frequent_itemsets,metric="lift",min_threshold=0.7)
rules


# In[10]:


rules.sort_values('lift',ascending=False)[0:20]


# In[11]:


rules[rules.lift>1] 


# In[12]:


plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


# In[13]:


plt.scatter(rules['support'], rules['lift'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs Lift’')
plt.show()


# In[14]:


plt.scatter(rules['lift'], rules['confidence'], alpha=0.5)
plt.xlabel('lift')
plt.ylabel('confidence')
plt.title(' Lift vs Confidence')
plt.show()


# In[15]:


fit = np.polyfit(rules['lift'], rules['confidence'], 1)
fit_fn = np.poly1d(fit)
plt.plot(rules['lift'], rules['confidence'], 'yo', rules['lift'], 
 fit_fn(rules['lift']))


# In[16]:


plt.bar(x = list(range(1,11)),height = frequent_itemsets.support[1:11],color='rgmyk');
plt.xticks(list(range(1,11)),frequent_itemsets.itemsets[1:11])
plt.xlabel('item-sets')
plt.ylabel('support')


# In[19]:


#considering the same support value and threshold value
frequent_itemsets1=apriori(dummy,min_support=0.3,use_colnames=True)
frequent_itemsets1
rules=association_rules(frequent_itemsets1,metric="lift",min_threshold=0.8)
rules
rules.sort_values('lift',ascending=False)[0:20]
rules[rules.lift>2] #f we change lift value to 2 then we get only one data 


# In[ ]:




