#!/usr/bin/env python
# coding: utf-8

# In[45]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


# In[15]:


data = pd.read_csv('C:\\Users\\Zara\\Documents\\ANALISIS-AP\\alumnos\\Zara-UG\\crime_data.csv',sep = ',',header = None)


# In[22]:


type(data)


# In[23]:


print(data)


# In[47]:


def distancia(x,c):
#calcula la distancia entre el punto x y la cámara c
#x y c son vectores que tienen en la primera entrada la latitud y en la segunda la longitud del punto
    lat_x = x[0]
    lon_x = x[1]
    lat_c = c[0]
    lon_c = c[1]
    R = 6373
    lat_x = math.radians(lat_x)
    lon_x = math.radians(lon_x)
    lat_c = math.radians(lat_c)
    lon_c = math.radians(lon_c)
    a = math.sin(lat_c - lat_x / 2)**2 + math.cos(lat_x) * math.cos(lat_c) * math.sin(lon_c - lon_x / 2)**2
    b = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R*b


# In[ ]:




