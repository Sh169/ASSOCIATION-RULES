#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt


# In[9]:


book=pd.read_csv("book (1).csv")
book.head()


# In[13]:


frequent_itemsets = apriori(book, min_support=0.1, use_colnames=True)
frequent_itemsets


# In[42]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.7)
rules


# In[43]:


rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
rules


# In[44]:


rules[ (rules['antecedent_len'] >= 2) &
       (rules['confidence'] > 0.7) &
       (rules['lift'] > 1) ]


# ### Visualizstion Rules

# In[20]:


plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


# ### Support vs Lift

# In[21]:


plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


# ### Lift vs Confidence

# In[22]:


plt.scatter(rules['lift'], rules['confidence'], alpha=0.5)
plt.xlabel('lift')
plt.ylabel('confidence')
plt.title(' Lift vs Confidence')
plt.show()


# In[23]:


fit = np.polyfit(rules['lift'], rules['confidence'], 1)
fit_fn = np.poly1d(fit)
plt.plot(rules['lift'], rules['confidence'], 'yo', rules['lift'], 
 fit_fn(rules['lift']))


# ### Bar Plot

# In[24]:


plt.bar(x = list(range(1,11)),height = frequent_itemsets.support[1:11],color='rgmyk');
plt.xticks(list(range(1,11)),frequent_itemsets.itemsets[1:11])
plt.xlabel('item-sets')
plt.ylabel('support')


# In[26]:


#with different values
frequent_itemsets1 = apriori(book, min_support=0.2, use_colnames=True)
frequent_itemsets1


# In[27]:


rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
rules


# In[37]:


#By reducing the values
rules[ (rules['antecedent_len'] >=1) &
       (rules['confidence'] > 0.6) &
       (rules['lift'] > 0.005) ]


# In[38]:


plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


# In[39]:


plt.scatter(rules['lift'], rules['confidence'], alpha=0.5)
plt.xlabel('lift')
plt.ylabel('confidence')
plt.title(' Lift vs Confidence')
plt.show()


# In[40]:


fit = np.polyfit(rules['lift'], rules['confidence'], 1)
fit_fn = np.poly1d(fit)
plt.plot(rules['lift'], rules['confidence'], 'yo', rules['lift'], 
 fit_fn(rules['lift']))


# In[41]:


plt.bar(x = list(range(1,11)),height = frequent_itemsets.support[1:11],color='rgmyk');
plt.xticks(list(range(1,11)),frequent_itemsets.itemsets[1:11])
plt.xlabel('item-sets')
plt.ylabel('support')

