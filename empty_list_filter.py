#!/usr/bin/env python
# coding: utf-8

# # Empty List Filter

# This is a technique using [this keras classification open kernel](https://www.kaggle.com/mobassir/keras-efficientnetb2-for-classifying-cloud).
# 
# We didn't retrain the classifier, just clear some over confident predictions

# In[18]:

from argparse import ArgumentParser
import pandas as pd
import numpy as np


# In[19]:


ap = ArgumentParser()

ap.add_argument("--csv", dest = "csv", type = str, help = "csv file path")
args = ap.parse_args()

SUBFILE = args.csv

submission = pd.read_csv(SUBFILE)


# In[20]:


image_labels_empty = set(pd.read_csv("empty_list.csv")["Empty"])
print("Picture x class that suppose to be empty:\t%s"%(len(image_labels_empty)))


# Non Empety For Now

# In[21]:


predictions_nonempty = set(submission.loc[~submission['EncodedPixels'].isnull(), 'Image_Label'].values)
print("Mask we have:\t%s"%(len(predictions_nonempty)))


# In[22]:


print(f'{len(image_labels_empty.intersection(predictions_nonempty))} masks would be removed')


# In[25]:


#removing masks
submission.loc[submission['Image_Label'].isin(image_labels_empty), 'EncodedPixels'] = np.nan
SAVE_PATH = 'emp_%s'%(SUBFILE)
print("New submission saved to :\t%s"%(SAVE_PATH))
submission.to_csv(SAVE_PATH, index=None)


# In[ ]:




