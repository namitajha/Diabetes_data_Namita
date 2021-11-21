#!/usr/bin/env python
# coding: utf-8

# # <center><b>Analysis of Diabetes Dataset<b></center>

# # Table of contents
# **1.** [**Introduction**](#Section1)<br>
# **2.** [**Problem Statement**](#Section2)<br>
# **3.** [**Importing Libraries**](#Section3)<br>
# **4.** [**Data Acquisition & Description**](#Section4)<br>
# **5.** [**Data Pre-profiling**](#Section5)<br>
# **6.** [**Data Cleaning**](#Section6)<br>
# **7.** [**Data Post-profiling**](#Section7)<br>
# **8.** [**Exploratory Data Analysis**](#Section8)<br>
# **9.** [**Conclusion**](#Section9)<br>
# 

# 
# <a name = Section1></a>
# # **1. Introduction**
# This dataset represents 10 years (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks. It includes over 50 features representing patient and hospital outcomes. Information was extracted from the database for encounters that satisfied the following criteria.
# 
# 1. It is an inpatient encounter (a hospital admission).
# 2. It is a diabetic encounter, that is, one during which any kind of diabetes was entered to the system as a diagnosis.
# 3. The length of stay was at least 1 day and at most 14 days.
# 4. Laboratory tests were performed during the encounter.
# 5. Medications were administered during the encounter.
# 
# The data contains attributes such as patient number, race, gender, age, admission type, time in hospital, medical specialty of admitting physician, number of lab test performed, HbA1c test result, diagnosis, number of medication, diabetic medications, number of outpatient, inpatient, and emergency visits in the year before the hospitalization, etc.

# ![diabetes%20image.jpg](attachment:diabetes%20image.jpg)

# ## Feature name and their description - 
# 
# 1. Encounter ID - Unique identifier of an encounter 
# 2. Patient number - Unique identifier of a patient 
# 3. Race - Values: Caucasian, Asian, African American, Hispanic, and other 
# 4. Gender - Values: male, female, and unknown/invalid 
# 5. Age - Grouped in 10-year intervals: 0, 10), 10, 20), …, 90, 100) 
# 6. Weight - Weight in pounds 
# 7. Admission type - Integer identifier corresponding to 9 distinct values, for example, emergency, urgent, elective, newborn, and not available | 
# 8. Discharge disposition - Integer identifier corresponding to 29 distinct values, for example, discharged to home, expired, and not availabl  
# 9. Admission source - Integer identifier corresponding to 21 distinct values, for example, physician referral, emergency room, and transfer from a hospital  
# 10. Time in hospital - Integer number of days between admission and discharge 
# 11. Payer code - Integer identifier corresponding to 23 distinct values, for example, Blue Cross/Blue Shield, Medicare, and self-pay
# 12. Medical specialty - Integer identifier of a specialty of the admitting physician, corresponding to 84 distinct values, for example, cardiology, internal medicine, family/general practice, and surgeon 
# 13. Number of lab procedures - Number of lab tests performed during the encounter 
# 14. Number of procedures - Number of procedures (other than lab tests) performed during the encounter 
# 16. Number of medications - Number of distinct generic names administered during the encounter 
# 17. Number of outpatient visits - Number of outpatient visits of the patient in the year preceding the encounter 
# 18. Number of emergency visits - Number of emergency visits of the patient in the year preceding the encounter
# 19. Number of inpatient visits - Number of inpatient visits of the patient in the year preceding the encounter
# 20. Diagnosis 1 - The primary diagnosis (coded as first three digits of ICD9); 848 distinct values
# 21. Diagnosis 2	- Secondary diagnosis (coded as first three digits of ICD9); 923 distinct values
# 22. Diagnosis 3 - Additional secondary diagnosis (coded as first three digits of ICD9); 954 distinct values
# 23. Number of diagnoses - Number of diagnoses entered to the system
# 24. Glucose serum test result - Indicates the range of the result or if the test was not taken. Values: “>200,” “>300,” “normal,” and “none” if not measured
# 25. A1c test result - Indicates the range of the result or if the test was not taken. Values: “>8” if the result was greater than 8%, “>7” if the result was greater than 7% but less than 8%, “normal” if the result was less than 7%, and “none” if not measured.
# 26. Change of medications - Indicates if there was a change in diabetic medications (either dosage or generic name). Values: “change” and “no change”
# 27. Diabetes medications - Indicates if there was any diabetic medication prescribed. Values: “yes” and “no”
# 28. 24 features for medications	 - For the generic names: metformin, repaglinide, nateglinide, chlorpropamide, glimepiride, acetohexamide, glipizide, glyburide, tolbutamide, pioglitazone, rosiglitazone, acarbose, miglitol, troglitazone, tolazamide, examide, sitagliptin, insulin, glyburide-metformin, glipizide-metformin, glimepiride-pioglitazone, metformin-rosiglitazone, and metformin-pioglitazone, the feature indicates whether the drug was prescribed or there was a change in the dosage. Values: “up” if the dosage was increased during the encounter, “down” if the dosage was decreased, “steady” if the dosage did not change, and “no” if the drug was not prescribed
# 29. Readmitted - Days to inpatient readmission. Values: “<30” if the patient was readmitted in less than 30 days, “>30” if the patient was readmitted in more than 30 days, and “No” for no record of readmission.

# <a name = Section2></a>
# # **2. Problem Statement**

# ### To study the factors influencing hospital readmission rates

# <a name = Section3></a>
# # **3. Importing Libraries**

# In[1]:


import numpy as np
import pandas as pd
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt


# <a name = Section4></a>
# # **4. Data Acquisition & Wrangling**

# ## 4.1 Reading data from a csv file 

# In[2]:


data = pd.read_csv(filepath_or_buffer='C:/Users/dell/Documents/diabetic_data.csv')
pd.options.display.max_columns = None


# In[3]:


data


# ## 4.2 Data description 
# ### Here, we will explore the data a little bit.

# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.head(3)


# ### We observed that though there is no missing value in the dataset, instead '?' is placed where there is an unkown value.

# <a name = Section5></a>
# 
# ---
# # **5. Data Pre-Profiling**
# ### Here we will see how many columns have "?" and its percentage. 

# In[8]:


data['diag_2'].value_counts(normalize=True)*100


# In[9]:


data['race'].value_counts(normalize=True)*100


# In[10]:


data['gender'].value_counts(normalize=True)*100


# In[11]:


data['age'].value_counts(normalize=True)*100


# In[12]:



data['weight'].value_counts(normalize=True)*100


# <a name = Section6></a>
# 
# ---
# # **6. Data Cleaning**
# ### We will start cleansing process with removing those fields which are not relevant to our objective or we do not have enough information about them. We decided to remove field of weight also though it is an important field, as 97% of the values are missing.
#  

# In[13]:


data_new = data.drop(['encounter_id', 'weight', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'payer_code', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient'], axis = 1) 


# In[14]:


data_new


# ### We observed that in whole dataset, we have '?' where we have a missing value. We will replace '?' with 'unknown'.

# In[41]:


data_new = data_new.replace(to_replace = '?', value = 'Unknown')
data_new


# <a name = Section7></a>
# 
# ---
# # **7. Data Post-Profiling**
# ### Here, we will observe the changes performed during data cleansing process.

# In[16]:


data_new.info()


# In[17]:


data_new['diag_2'].value_counts(normalize=True)*100


# In[18]:


data_new['race'].value_counts(normalize=True)*100


# In[19]:


data_new['gender'].value_counts(normalize=True)*100


# ### We observed that unnecessary fields have been removed and missinf values have been replaced by "Unknown".

# <a name = Section8></a>
# 
# ---
# # **8. Exploratory Data Analysis**

# ## 8.1 Here, we will see demographic distribution of the data.
# ### 8.1.1 Gender distibution of the data

# In[42]:


fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
Gender = data_new['gender'].value_counts().index
Values = data_new['gender'].value_counts(normalize=True).values*100

plt.bar(Gender, Values, color ='Orange',
        width = 0.4)
 
plt.xlabel("gender")
plt.ylabel("%age")
plt.title("Gender distribution of the data")
plt.show()


# ### 8.1.2 Race distibution in the data

# In[43]:


fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
Race = data_new['race'].value_counts().index
Values = data_new['race'].value_counts(normalize=True).values*100

plt.bar(Race, Values, color ='purple',
        width = 0.4)
 
plt.xlabel("Race")
plt.ylabel("%age")
plt.title("Race distribution of the data")
plt.show()


# ### 8.1.3 Age distibution of the data

# In[47]:


fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
Age = data_new['age'].value_counts().index
Values = data_new['age'].value_counts(normalize=True).values*100

plt.bar(Age, Values, color ='Maroon',
        width = 0.4)
 
plt.xlabel("Age")
plt.ylabel("%age")
plt.title("Age distribution of the data")
plt.show()


# ### 8.1.3 Percentage of patients with HbA1c tested

# In[23]:


fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
HbA1c = data_new['A1Cresult'].value_counts().index
Values = data_new['A1Cresult'].value_counts(normalize=True).values*100

plt.bar(HbA1c, Values,
        width = 0.4)
 
plt.xlabel("%age of patients with HbA1c tested")
plt.ylabel("%age")
plt.title("Patients with HbA1c tested")
plt.show()


# ### We observed from the above graphs that in the dataset:
# - gender ratio of the dataset is almost equal
# - Ratio of Caucasians was highest
# - Ratio of elderly patients is higher
# - More than 80% of patients were not tested for HbA1c

# ## 8.2 Creating stacked bar graph for finding relationship of readmission with different variables
# ### 8.2.1 Age v/s Readmission

# In[24]:


data_new.groupby(['age', 'readmitted'])[['readmitted']].size().unstack(fill_value=0).plot(kind='bar', stacked=True)
plt.xlabel("Age in Years")
plt.ylabel("Readmissions")
plt.title("Age v/s Readmission")
plt.show()


# ### The stacked bar graph shows that readmissions increases till the age of 70-80 and after that it starts decreasing.

# ### 8.2.2 HbA1c test v/s Readmissions

# In[48]:


data_new.groupby(['A1Cresult', 'readmitted'])[['readmitted']].size().unstack(fill_value=0).plot(kind='bar', stacked=True)
plt.xlabel("HbA1c test")
plt.ylabel("Readmissions")
plt.title("HbA1c test v/s Readmission")
plt.show()


# ### Here, we can leave the 'None' category where no test was ordered and our 80% population lies there. Leaving "None", there were highest readmissions observed where glycemic index was higher than 8, which makes sense.

# ### 8.3 Next we wanted to see how many patients were tested for HbA1c when there was a primary diagnosis of diabetes who were readmitted.
# 

# In[39]:


data_diab = data_new[data_new['diag_1'].str.contains(pat = '250')]
data_readmitted = data_diab[~data_diab['readmitted'].str.contains('NO')]

HbA1c = data_readmitted['A1Cresult'].value_counts().index
Values = data_readmitted['A1Cresult'].value_counts(normalize=True).values*100

plt.bar(HbA1c, Values,
        width = 0.4)
 
plt.xlabel("%age of patients with HbA1c tested")
plt.ylabel("%age")
plt.title("Readmitted patients with HbA1c tested when there was a primary diagnosis of diabetes")
plt.show()


# ### Here, it was quite interesting to know that even though there was a primary diagnosis of diabetes, around 72% of the patients were not ordered for HbA1c test and were readmitted. 

# ### 8.4 For how many patients there was a change in medication when HbA1c test was done

# In[40]:


data_change = data_new[data_new['change'].str.contains(pat = 'Ch')]

HbA1c = data_diab['A1Cresult'].value_counts().index
Values = data_diab['A1Cresult'].value_counts(normalize=True).values*100

plt.bar(HbA1c, Values,
        width = 0.4)
 
plt.xlabel("%age of patients with HbA1c tested")
plt.ylabel("%age")
plt.title("Patients with HbA1c tested when there was a change in medication")
plt.show()


# ### The graph shows that there was frequent change in medication when glycemix index was higher than 8.

# <a name = Section9></a>
# 
# ---
# # **9. Conclusion**
# ---
# - Results show that the measurement of HbA1c was performed infrequently (18.4%) in the inpatient setting.
# - When HbA1c test was ordered, and glycemic index was higher than 8%, there was more change in medication.
# - The result suggest that more testing of HbA1c may lead to better medication, and thus less future readmissions and less cost of care for diabetic patients. 
# 
# 

# In[ ]:




