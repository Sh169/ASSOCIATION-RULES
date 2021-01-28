#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt


# In[2]:


movie=pd.read_csv("my_movies.csv")
movie.head()


# In[3]:


dummy=pd.get_dummies(movie[["V1","V2","V3","V4","V5"]])
dummy.head()


# In[4]:


frequent_itemsets=apriori(dummy,min_support=0.1,use_colnames=True)
frequent_itemsets


# In[8]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.7)
rules


# In[10]:


rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
rules


# In[11]:


rules[ (rules['antecedent_len'] >= 2) &
       (rules['confidence'] > 0.7) &
       (rules['lift'] > 1) ]


# ### Visualize the data

# In[12]:


plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


# In[13]:


plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


# In[15]:


plt.scatter(rules['lift'], rules['confidence'], alpha=0.5)
plt.xlabel('lift')
plt.ylabel('confidence')
plt.title(' Lift vs Confidence')
plt.show()


# In[16]:


fit = np.polyfit(rules['lift'], rules['confidence'], 1)
fit_fn = np.poly1d(fit)
plt.plot(rules['lift'], rules['confidence'], 'yo', rules['lift'], 
 fit_fn(rules['lift']))


# In[19]:


plt.bar(x = list(range(1,11)),height = frequent_itemsets.support[1:11],color='rgmyk');
plt.xticks(list(range(1,11)),frequent_itemsets.itemsets[1:11])
plt.xlabel('item-sets')
plt.ylabel('support')


# In[28]:


#with different value 
frequent_itemsets1 = apriori(dummy, min_support=0.3, use_colnames=True)
frequent_itemsets1


# In[22]:


rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
rules


# In[30]:


#By changing the values
rules[ (rules['antecedent_len'] >=2) &
       (rules['confidence'] > 0.8) &
       (rules['lift'] > 1.2) ]


# In[24]:


plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


# In[25]:


plt.scatter(rules['lift'], rules['confidence'], alpha=0.5)
plt.xlabel('lift')
plt.ylabel('confidence')
plt.title(' Lift vs Confidence')
plt.show()


# In[26]:


fit = np.polyfit(rules['lift'], rules['confidence'], 1)
fit_fn = np.poly1d(fit)
plt.plot(rules['lift'], rules['confidence'], 'yo', rules['lift'], 
 fit_fn(rules['lift']))


# In[27]:


plt.bar(x = list(range(1,11)),height = frequent_itemsets.support[1:11],color='rgmyk');
plt.xticks(list(range(1,11)),frequent_itemsets.itemsets[1:11])
plt.xlabel('item-sets')
plt.ylabel('support')

