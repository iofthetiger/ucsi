#!/usr/bin/env python
# coding: utf-8

# # Empty List Filter

# This is a technique using [this keras classification open kernel](https://www.kaggle.com/mobassir/keras-efficientnetb2-for-classifying-cloud).
# 
# We didn't retrain the classifier, just clear some over confident predictions

# In[1]:


import pandas as pd
import numpy as np
from argparse import ArgumentParser


# In[3]:


ap = ArgumentParser()

ap.add_argument("--csv", dest = "csv", type = str, help = "csv file path")
args = ap.parse_args()

SUBFILE = args.csv

# SUBFILE = "convex_csvs_1117_012120_sub.csv"

submission = pd.read_csv(SUBFILE)


# In[10]:


sub_ = pd.read_csv(SUBFILE)
sub_["img"] = sub_.Image_Label.apply(lambda x:x.split("_")[0])
sub_["lbl"] = sub_.Image_Label.apply(lambda x:x.split("_")[1])
sub_["has_class"] = sub_.EncodedPixels.isnull()*1


# In[14]:


class_counter = sub_.groupby("img").sum().reset_index()


# In[18]:


missing = class_counter[class_counter.has_class==0]


# In[20]:


img2counter = dict(zip(class_counter["img"],class_counter["has_class"]))


# In[22]:


sub_["counter"] = sub_.img.apply(lambda x: img2counter[x])


# In[23]:


sub_


# In[24]:


image_labels_empty = set(pd.read_csv("empty_list.csv")["Empty"])
print("Picture x class that suppose to be empty:\t%s"%(len(image_labels_empty)))


# Non Empety For Now

# In[41]:


predictions_nonempty = set(sub_.loc[~sub_['EncodedPixels'].isnull(), 'Image_Label'].values)
nonempty_single = set(sub_.loc[((~sub_['EncodedPixels'].isnull())& sub_.counter==1), 'Image_Label'].values)
print("Mask we have:\t%s"%(len(predictions_nonempty)))
print("Non empty single %s"%(len(nonempty_single)))


# In[43]:


nonempty_can_remove = predictions_nonempty-nonempty_single


# In[44]:


print(f'{len(image_labels_empty.intersection(predictions_nonempty))} masks are in intersection')
print(f'{len(image_labels_empty.intersection(nonempty_single))} masks are in intersection & single')
print(f'{len(image_labels_empty.intersection(nonempty_can_remove))} masks will be removed')


# In[45]:

remove_list = image_labels_empty.intersection(nonempty_can_remove)

print(remove_list)
#removing masks
submission.loc[submission['Image_Label'].isin(remove_list), 'EncodedPixels'] = np.nan
SAVE_PATH = 'emp_%s'%(SUBFILE)
print("New submission saved to :\t%s"%(SAVE_PATH))
submission.to_csv(SAVE_PATH, index=None)


# In[ ]:




