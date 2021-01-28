#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt


# In[2]:


book=pd.read_csv("book (1).csv")
book.head()


# ### Apriori Algorithm

# In[3]:


#Considering the support value as 0.1
frequent_itemsets=apriori(book,min_support=0.1,use_colnames=True)
frequent_itemsets


# In[4]:


rules=association_rules(frequent_itemsets,metric="lift",min_threshold=0.7)
rules


# In[5]:


rules.sort_values('lift',ascending=False)[0:20]


# In[6]:


rules[rules.lift>1] 


# ### Visualizing results

# In[7]:


plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


# ### Support vs lift

# In[8]:


plt.scatter(rules['support'], rules['lift'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs Lift’')
plt.show()


# ### Lift vs Confidence

# In[9]:


plt.scatter(rules['lift'], rules['confidence'], alpha=0.5)
plt.xlabel('lift')
plt.ylabel('confidence')
plt.title(' Lift vs Confidence')
plt.show()


# In[10]:


fit = np.polyfit(rules['lift'], rules['confidence'], 1)
fit_fn = np.poly1d(fit)
plt.plot(rules['lift'], rules['confidence'], 'yo', rules['lift'], 
 fit_fn(rules['lift']))


# In[11]:


plt.bar(x = list(range(1,11)),height = frequent_itemsets.support[1:11],color='rgmyk');
plt.xticks(list(range(1,11)),frequent_itemsets.itemsets[1:11])
plt.xlabel('item-sets')
plt.ylabel('support')


# In[12]:


#considering the same support value and threshold value
frequent_itemsets1=apriori(book,min_support=0.1,use_colnames=True)
frequent_itemsets1
rules=association_rules(frequent_itemsets1,metric="lift",min_threshold=0.7)
rules
rules.sort_values('lift',ascending=False)[0:20]
rules[rules.lift>2] #f we change lift value to 2 then we get only one data 


# In[ ]:





# In[ ]:





# In[ ]:




