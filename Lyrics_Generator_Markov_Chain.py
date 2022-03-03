#!/usr/bin/env python
# coding: utf-8

# # Lyrics Generator Code using Markov Chain

# In[22]:


import numpy as np
np.random.seed(11)


# In[23]:


def generateTable(data,k=4):
    T = {}
    for i in range(len(data)-k):
        X = data[i:i+k]
        Y = data[i+k]
        #print("X  %s and Y %s  "%(X,Y))
        
        if T.get(X) is None:
            T[X] = {}
            T[X][Y] = 1
        else:
            if T[X].get(Y) is None:
                T[X][Y] = 1
            else:
                T[X][Y] += 1
    
    return T


# In[24]:


def convertFreqIntoProb(T):     
    for kx in T.keys():
        s = float(sum(T[kx].values()))
        for k in T[kx].keys():
            T[kx][k] = T[kx][k]/s
                
    return T


# In[25]:


# T = generateTable("Coding is love . Vivek Rai Coder")
# print(T)


# In[26]:


# T = convertFreqIntoProb(T)
# print(T)


# In[27]:


text_path = "Apna Time Aayega.txt"
def load_text(filename):
    with open(filename,encoding='utf8') as f:
        return f.read().lower()
    
text = load_text(text_path)


# In[28]:


# print(text)


# In[29]:


def trainMarkovChain(text,k=4):
    
    T = generateTable(text,k)
    T = convertFreqIntoProb(T)
    
    return T


# In[30]:


model = trainMarkovChain(text)


# In[31]:


# print(model)


# In[32]:


def sample_next(ctx,T,k):
    ctx = ctx[-k:]
    if T.get(ctx) is None:
        return " "
    possible_Chars = list(T[ctx].keys())
    possible_values = list(T[ctx].values())
    
    #print(possible_Chars)
    #print(possible_values)
    
    return np.random.choice(possible_Chars,p=possible_values)


# In[33]:


def generateText(starting_sent,k=4,maxLen=1000):
    
    sentence = starting_sent
    ctx = starting_sent[-k:]
    
    for ix in range(maxLen):
        next_prediction = sample_next(ctx,model,k)
        sentence += next_prediction
        ctx = sentence[-k:]
    return sentence


# In[36]:


text = generateText("apna",k=4,maxLen=2000)
print(text)


# In[37]:


sub = open("mySub.txt","w",encoding='utf8')
sub.write(text) 
sub.close()


# In[ ]:




