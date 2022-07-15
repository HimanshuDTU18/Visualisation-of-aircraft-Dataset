#!/usr/bin/env python
# coding: utf-8

# # Importing the packages

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


np.set_printoptions(suppress=  True , linewidth = 100 , precision = 2 )


# # Importing the Data

# In[3]:


raw_data_np = np.genfromtxt('loan-data.csv' ,delimiter=';' ,skip_header = 1 )
raw_data_np


# # Checking for Incomplete Data

# In[4]:


np.isnan(raw_data_np ).sum()


# In[5]:


temp_fill = np.nanmax(raw_data_np)+1
temp_mean = np.nanmean(raw_data_np , axis = 0)


# In[6]:


temp_mean


# In[7]:


temp_stats = np.array([np.nanmin(raw_data_np , axis = 0),temp_mean , np.nanmax(raw_data_np , axis = 0)])


# In[8]:


temp_stats


# # Splitting the Dataset
# 
# ## Splitting the columns

# In[9]:


columns_strings = np.argwhere(np.isnan(temp_mean)).squeeze()
columns_strings


# In[10]:


columns_numeric = np.argwhere(np.isnan(temp_mean)==False).squeeze()
columns_numeric


# # Re - importing the dataset
# 

# In[11]:


loan_data_strings = np.genfromtxt("loan-data.csv",
                                 delimiter = ';',
                                 skip_header = 1,
                                  autostrip = True,
                                 usecols = columns_strings , 
                                 dtype = str)
loan_data_strings


# In[12]:


loan_data_numeric = np.genfromtxt("loan-data.csv",
                                 delimiter = ';',
                                 skip_header = 1,
                                  autostrip = True,
                                 usecols = columns_numeric , 
                                filling_values = temp_fill)
loan_data_numeric


# # The Names of the Columns
# 

# In[13]:


header_full = np.genfromtxt("loan-data.csv",
                                 delimiter = ';',
                                 skip_footer = raw_data_np.shape[0],
                                   
                                 dtype = str)
header_full


# In[14]:


header_strings , header_numeric = header_full[columns_strings] , header_full[columns_numeric]


# In[15]:


header_strings


# In[16]:


header_numeric


# # Manipulating String Columns

# In[17]:


header_strings


# In[18]:


header_strings[0]='isuue_date'


# In[19]:


header_strings


# ### issue Date

# In[20]:


np.unique(loan_data_strings[:,0])


# In[21]:


loan_data_strings[:, 0 ]= np.chararray.strip(loan_data_strings[:,0] , '-15')


# In[22]:


loan_data_strings[:,0]


# In[23]:


np.unique(loan_data_strings[:,0])


# In[24]:


months = np.array(['','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])


# In[25]:


for i in range(13):
    loan_data_strings[:,0] = np.where(loan_data_strings[:,0]==months[i],
                                     i,
                                     loan_data_strings[:,0])


# In[26]:


np.unique(loan_data_strings[:,0])


# In[27]:


loan_data_strings[:,0]


# ### Loan Status

# In[28]:


header_strings


# In[29]:


np.unique(loan_data_strings[:,1])


# In[30]:


np.unique(loan_data_strings[:,1]).size


# In[31]:


status_bad = np.array(["" ,"Charged off" ,"default" ,"Late(31-120 days)"])


# In[32]:


loan_data_strings[:,1]=np.where(np.isin(loan_data_strings[:,1],status_bad),0,1)


# In[33]:


np.unique(loan_data_strings[:,1])


# ### Term

# In[34]:


header_strings


# In[35]:


np.unique(loan_data_strings[:,2])


# In[36]:


loan_data_strings[:,2] = np.chararray.strip(loan_data_strings[:,2]," months")


# In[37]:


loan_data_strings[:,2]


# In[38]:


header_strings[2] = "term_months"


# In[39]:


loan_data_strings[:,2] = np.where(loan_data_strings[:,2]=="",
                                 '60',
                                 loan_data_strings[:,2])


# In[40]:


np.unique(loan_data_strings[:,2])


# ### Grade and subgrade

# In[41]:


header_strings


# In[42]:


np.unique(loan_data_strings[:,3])


# In[43]:


np.unique(loan_data_strings[:,4])


# ### Filling sub grade

# In[44]:


for i in np.unique(loan_data_strings[:,3])[1:]:
    loan_data_strings[:,4] = np.where((loan_data_strings[:,4]=="")&(loan_data_strings[:,3]==i),
                                     i+'5',
                                     loan_data_strings[:,4])
    


# In[45]:


np.unique(loan_data_strings[:,4],return_counts = True)


# In[46]:


loan_data_strings[:,4] = np.where((loan_data_strings[:,4]==""),
                                    'H1',
                                    loan_data_strings[:,4])


# In[47]:


np.unique(loan_data_strings[:,4])


# ### Removing Grade

# In[48]:


loan_data_strings = np.delete( loan_data_strings ,3 , axis = 1)


# In[49]:


loan_data_strings[:,3]


# In[50]:


header_strings = np.delete(header_strings ,3 )


# In[51]:


header_strings[3]


# ### Converting Sub Grade

# In[52]:


keys = list(np.unique(loan_data_strings[:,3]))
values = list(range(1 , np.unique(loan_data_strings[:,3]).shape[0]+1))
dict_sub_grade = dict(zip(keys ,values))


# In[53]:


for i in np.unique(loan_data_strings[:,3]):
    loan_data_strings[:,3] = np.where(loan_data_strings[:,3]==i,
                                     dict_sub_grade[i],
                                       loan_data_strings[:,3] )


# In[54]:


np.unique(loan_data_strings[:,3])


# ### Verification Status

# In[55]:


np.unique(loan_data_strings[:,4])


# In[56]:


loan_data_strings[:,4] = np.where((loan_data_strings[:,4]=="")|(loan_data_strings[:,4]=="Not Verified"),0,1)


# In[57]:


np.unique(loan_data_strings[:,4])


# In[58]:


np.unique(loan_data_strings[:,5])


# In[59]:


loan_data_strings[:,5]=np.chararray.strip(loan_data_strings[:,5] ,"https://www.lendingclub.com/browse/loanDetail.action?loan_id=")


# In[60]:


header_full


# In[61]:


loan_data_strings = np.delete(loan_data_strings,5 , axis = 1 )
header_strings = np.delete(header_strings ,5)


# In[62]:


loan_data_strings[:,5]


# In[63]:


header_strings[5]


# ### State Address

# In[64]:


header_strings


# In[65]:


header_strings[5]="State_Address"


# In[66]:


states_names , states_count =np.unique(loan_data_strings[:,5],return_counts=True)
states_count_sorted = np.argsort(-states_count)
states_names[states_count_sorted] ,states_count[states_count_sorted]


# In[67]:


states_west = np.array(['WA','OR','CA','NV','ID','MT','WY','UT','CO','AZ','NM','HI','AK'])
states_south = np.array(['TX','OK','AR','LA','MS','AL','TN','KY','FL','GA','SC','NC','VA','WV','MD','DE','DC'])
states_midwest = np.array(['ND','SD','NE','KS','MN','IA','MO','WI','IL','IN','MI','OH'])
states_east = np.array(['PA','NY','NJ','CT','MA','VT','NH','ME','RI'])


# In[68]:


loan_data_strings[:,5] = np.where(loan_data_strings[:,5]=="",
                                 0,
                                 loan_data_strings[:,5])


# In[69]:


loan_data_strings[:,5] =np.where(np.isin(loan_data_strings[:,5],states_west),1,loan_data_strings[:,5])
loan_data_strings[:,5] =np.where(np.isin(loan_data_strings[:,5],states_south),2,loan_data_strings[:,5])
loan_data_strings[:,5] =np.where(np.isin(loan_data_strings[:,5],states_midwest),3,loan_data_strings[:,5])
loan_data_strings[:,5] =np.where(np.isin(loan_data_strings[:,5],states_east),4,loan_data_strings[:,5])


# In[70]:


np.unique(loan_data_strings[:,5])


# ### Converting to Numbers

# In[71]:


loan_data_strings = loan_data_strings.astype(np.int)


# In[72]:


loan_data_strings


# # Manipulating Numeric Columns

# In[73]:


loan_data_numeric


# In[74]:


np.isnan(loan_data_numeric).sum()


# ## Substite 'Filler' values

# In[75]:


header_numeric


# ### Id

# In[76]:


temp_fill


# In[77]:


np.isin(loan_data_numeric[:,0],temp_fill).sum()


# ### temp stats

# In[78]:


temp_stats[:,columns_numeric]


# ### Funded Amount

# In[79]:


loan_data_numeric[:,2]


# In[80]:


loan_data_numeric[:,2]=np.where(loan_data_numeric[:,2]==temp_fill,
                               temp_stats[0 , columns_numeric[2]],
                               loan_data_numeric[:,2])
loan_data_numeric[:,2]


# ### Loaned Amount , Intrest Rate , Total Payment,Installment

# In[81]:


header_numeric


# In[82]:


for i in[1,3,4,5]:
    loan_data_numeric[:,i] = np.where(loan_data_numeric[:,i]==temp_fill,
                                    temp_stats[2,columns_numeric[i]],
                                    loan_data_numeric[:,i])
    


# In[83]:


loan_data_numeric


# # Currency Exchange

# ### The Exchange Rate

# In[84]:


EUR_USD = np.genfromtxt("EUR-USD.csv",delimiter = ',' ,autostrip = True ,skip_header=1 , usecols =3)


# In[85]:


EUR_USD


# In[86]:


loan_data_strings[:,0]


# In[87]:


exchange_rate = loan_data_strings[:,0]

for i in range(1,13):
    exchange_rate = np.where(exchange_rate == i ,
                            EUR_USD[i-1],
                            exchange_rate)

exchange_rate = np.where(exchange_rate == 0,
                        np.mean(EUR_USD),
                        exchange_rate)

exchange_rate


# In[88]:


exchange_rate = np.reshape(exchange_rate , (10000,1))


# In[89]:


loan_data_numeric = np.hstack((loan_data_numeric , exchange_rate))


# In[90]:


header_numeric = np.concatenate((header_numeric , np.array(['exchange_rate'])))
header_numeric


# ## USD - EUR

# In[91]:


header_numeric


# In[92]:


columns_dollar = np.array([1,2,4,5])


# In[93]:


loan_data_numeric[:,6]


# In[94]:


for i in columns_dollar:
    loan_data_numeric = np.hstack((loan_data_numeric , np.reshape(loan_data_numeric[:,i] /loan_data_numeric[:,6],(10000,1))))


# In[95]:


loan_data_numeric.shape


# ### Expanding Array 

# In[96]:


header_additional = np.array([column_name +'_EUR' for column_name in header_numeric[columns_dollar]])


# In[97]:


header_additional


# In[98]:


header_numeric = np.concatenate((header_numeric, header_additional))


# In[99]:


header_numeric [columns_dollar] = np.array([column_name +'USD' for column_name in header_numeric[columns_dollar]])


# In[100]:


header_numeric


# In[101]:


columns_index_order = [0,1,7,2,8,3,4,5,10,6]


# In[102]:


header_numeric = header_numeric[columns_index_order]


# In[103]:


loan_data_numeric


# In[104]:


loan_data_numeric = loan_data_numeric[:,columns_index_order]


# ### Interest Rate

# In[105]:


header_numeric


# In[106]:


loan_data_numeric[:,5]


# In[107]:


loan_data_numeric[:,5] = loan_data_numeric[:,5]/100


# In[108]:


loan_data_numeric[:,5]


# In[110]:


loan_data_numeric.shape


# In[112]:


loan_data_strings.shape


# In[113]:


loan_data = np.hstack((loan_data_strings, loan_data_numeric))


# In[114]:


loan_data


# In[115]:


np.isnan(loan_data).sum()


# In[116]:


header_full = np.concatenate((header_strings, header_numeric))


# In[117]:


header_full


# In[118]:


loan_data = loan_data[np.argsort(loan_data[:,0])]


# In[119]:


loan_data


# In[120]:


np.argsort(loan_data[:,0])


# ## Storing the New Dataset

# In[121]:


loan_data = np.vstack((header_full, loan_data))


# In[122]:


np.savetxt("loan-data-preprocessed.csv", 
           loan_data, 
           fmt = '%s',
           delimiter = ',')


# In[ ]:




