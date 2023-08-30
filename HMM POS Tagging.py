#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import operator
import json




df = pd.read_table('data/train', header=None)#,sep='\t',lineterminator='\n')
df.set_axis(["index", "word", "tag"], axis="columns", inplace=True)


# In[ ]:





# In[2]:


word_counts = {}

for index,row in df.iterrows():
    if row[1] in word_counts:
        word_counts[row[1]] += 1
    else:
        word_counts[row[1]] = 1
        


# In[3]:


threshold = 3
unkcount = 0
unkwords = []

for entry in word_counts:
    if word_counts[entry] <= threshold:
        unkwords.append(entry)
        unkcount = unkcount + word_counts[entry]
        
# replace words with less freq as <unk>

df.loc[df['word'] .isin(unkwords),'word'] = '<unk>'


# In[4]:


word_counts = dict( sorted(word_counts.items(), key=operator.itemgetter(1),reverse=True))


# In[5]:


i = 0

with open("vocab.txt", 'w') as f:
    f.write('<unk>' + '\\t' + str(i) + '\\t' + str(unkcount) + '\n')
    i +=1
    for entry in word_counts:
        if word_counts[entry] > threshold:
            f.write(entry + '\\t' + str(i) + '\\t' + str(word_counts[entry]) + '\n')
            i +=1
            
print("vocab size:",i)
print("total occurrences of the special token ‘< unk >’:",unkcount)


# In[6]:


pos = df.tag.unique()


# In[7]:


transition = {}
deno = {}
num_transition = {}

for postag in pos:
    deno[postag] = df['tag'].value_counts()[postag]

for i in range(0, len(df)-1):
    if (df.tag[i],df.tag[i+1]) in num_transition:
        num_transition[(df.tag[i],df.tag[i+1])] += 1
    else:
        num_transition[(df.tag[i],df.tag[i+1])] = 1

for pair in num_transition:
    transition[pair[0],pair[1]] = num_transition[pair]/deno[pair[0]]
    
     


# In[8]:


emission = {}
num_emission = {}

for i in range(0, len(df)):
    if (df.tag[i],df.word[i]) in num_emission:
        num_emission[(df.tag[i],df.word[i])] += 1
    else:
        num_emission[(df.tag[i],df.word[i])] = 1

for pair1 in num_emission:
    emission[pair1[0],pair1[1]] = num_emission[pair1]/deno[pair1[0]]


       


# In[9]:


transition_json = {str(key): value for key, value in transition.items()}
emission_json = {str(key): value for key, value in emission.items()}



data = {"transition": transition_json, "emission": emission_json}

with open("hmm.json", "w") as file:
    json.dump(data, file)


# In[10]:


##transition mat

transitionmat = [[0 for j in range(len(pos))] for i in range(len(pos))]


poslist = pos.tolist()

for (i, j), value in transition.items():
    transitionmat[poslist.index(i)][poslist.index(j)] = value
    
print("number of transition parameters:", len(transitionmat[0])*len(transitionmat))

    
## emission mat
    
words = df.word.unique()
wordlist = words.tolist()

emissionmat = [[0 for j in range(len(words))] for i in range(len(pos))]

for (i, j), value in emission.items():
    emissionmat[poslist.index(i)][wordlist.index(j)] = value
    
print("number of emission parameters:", len(emissionmat[0])*len(emissionmat))

    


# In[11]:


import numpy as np

dfdev = pd.read_table('data/dev', header=None)#,sep='\t',lineterminator='\n')
dfdev.set_axis(["index", "word", "tag"], axis="columns", inplace=True)


### Greedy decoding HMM ###

initial = {}

for postag in pos:
    initial[postag] = deno[postag] / len(df)

tag_predicted = {}

for i in range(0, len(dfdev)): # for each word
    if(dfdev.word[i] in wordlist):
        curword = dfdev.word[i]
    else:
        curword = '<unk>'
    curidx = wordlist.index(curword)
    #print(curword)
    indexlist =[]
    for j in range(0, len(pos)): # for each tag
        
        if(dfdev['index'][i] != 1):
            indexlist.append(transitionmat[poslist.index(tag_predicted[i-1])][j] * emissionmat[j][curidx])
            
            
        else:
            indexlist.append(initial[pos[j]] * emissionmat[j][curidx])
    tag_predicted[i] = pos[np.argmax(indexlist)]
    
            
accuracycnt = 0
for i in range(0, len(dfdev)):
    if(dfdev.tag[i] == tag_predicted[i]):
        accuracycnt += 1
        
accuracy = accuracycnt/len(dfdev)
print("Accuracy for greedy on dev is",accuracy*100)  


        


# In[12]:


## runnning greedy on test data

dftest = pd.read_table('data/test', header=None)#,sep='\t',lineterminator='\n')
dftest.set_axis(["index", "word"], axis="columns", inplace=True)


### Greedy decoding HMM ###


tag_predicted = {}

for i in range(0, len(dftest)): # for each word
    if(dftest.word[i] in wordlist):
        curword = dftest.word[i]
    else:
        curword = '<unk>'
    curidx = wordlist.index(curword)
    #print(curword)
    indexlist =[]
    for j in range(0, len(pos)): # for each tag
        
        if(dftest['index'][i] != 1):
            indexlist.append(transitionmat[poslist.index(tag_predicted[i-1])][j] * emissionmat[j][curidx])
            
            
        else:
            indexlist.append(initial[pos[j]] * emissionmat[j][curidx])
    tag_predicted[i] = pos[np.argmax(indexlist)]


# In[23]:


## print value of greedy hmm decoding
i = 0
j = 1
with open("greedy.out", 'w') as f:
    for i in range(0, len(dftest)):
        
        
        
        if(dftest['index'][i]==1 and i!=0):
            f.write('\n')
            j=1
            f.write(str(j) + '\t' + dftest.word[i] + '\t' + str(tag_predicted[i]) + '\n')
            
            
        else:
            if(i!=0):
                j+=1
            f.write(str(j) + '\t' + dftest.word[i] + '\t' + str(tag_predicted[i]) + '\n')
            
        i +=1
f.close()


# In[14]:


## self Viterbi HMM ##


dfdev = pd.read_table('data/dev', header=None)#,sep='\t',lineterminator='\n')
dfdev.set_axis(["index", "word", "tag"], axis="columns", inplace=True)

prev_statelist = [] #memoizing the states
history = [0] * len(pos)
viterbi_final_taglist = []

#for i in range(0, len(dfdev)): #every word
for i in range(0, len(dfdev)):
    if dfdev.word[i] in wordlist:
        curword = dfdev.word[i]
        #print(curword)
    else:
        curword = '<unk>' 

    curidx=wordlist.index(curword) 

    #print(curidx)
    
    if(curword != '.'): #not end of sentence
    
        if(dfdev['index'][i] == 1):#first word of sentence
            #initial *  emission
            prev_prob = [0] * len(pos)
            prevstates = []

            for j in range(0, len(pos)): #was pos
                prev_prob[j]=initial[pos[j]] * emissionmat[j][curidx]
            prevstates = [-1] *len(pos) #wasnone
            prev_statelist.append(prevstates)
            history = prev_prob
            #print(prevstates)

        else: # subsequent word in sentence 
            # trans * emission * previous

            prev_prob = [0] * len(pos)
            prevstates = [-1] * len(pos) #was0

            for j in range(0, len(pos)): #was pos
                curr_max_prob = 0
                curr_state = -1

                for k in range(0,len(pos)): #was pos

                    prob = history[k] * transitionmat[k][j] * emissionmat[j][curidx]
                    #print(curword)
                    #print(curword,history[k],transitionmat[k][j],emissionmat[j][wordlist.index(curword)])
                    if prob > curr_max_prob:
                        curr_max_prob = prob
                        curr_state = k
                        #print(prob,k)
                prev_prob[j] = curr_max_prob
                prevstates[j] = curr_state

            history = prev_prob
            #print(prevstates)
            prev_statelist.append(prevstates)
        
    else: #end of sentence
        
#         prevstates[-1]* len(pos)
#         prevstates[10] = 1
#         prev_statelist.append(prevstates)
        
             
              
        viterbi_taglist = []
        viterbi_taglist.append(".")
        
        
        prev_idx = np.argmax(history)
        prev_tag = pos[prev_idx]
        viterbi_taglist.append(prev_tag)
        
#         l=len(prev_statelist)-2
        
#         while(prev_tag != None):
#             prev_idx = prev_statelist[l][prev_idx]
#             prev_tag = str(poslist[prev_idx])
#             viterbi_taglist.append(prev_tag)
#             l -=1
        for i in range(1,len(prev_statelist)):
            #print(i)
            prev_idx = prev_statelist[len(prev_statelist)-i][prev_idx]
            prev_tag = str(poslist[prev_idx])
            viterbi_taglist.append(prev_tag)

        #print(prev_statelist)
        viterbi_taglist.reverse()
        viterbi_final_taglist.append(viterbi_taglist)
            
        
        prev_statelist = [] #memoizing the states
        history = [0] * len(pos)
        
        #argax of history
        #assign index (argmax) = curr tag
        #pos[value of argmax] = prev tag






# In[15]:


viterbi_ans = []
for i in range(len(viterbi_final_taglist)):
    for j in range(len(viterbi_final_taglist[i])):
        viterbi_ans.append(viterbi_final_taglist[i][j])
        


# In[16]:


accuracycnt = 0
for i in range(0, len(dfdev)):
    if(dfdev.tag[i] == viterbi_ans[i]):
        accuracycnt += 1
        
accuracy = accuracycnt/len(dfdev)
print("Accuracy for viterbi on dev is",accuracy*100)


# In[17]:


## runnning viterbi on test data

dftest = pd.read_table('data/test', header=None)#,sep='\t',lineterminator='\n')
dftest.set_axis(["index", "word"], axis="columns", inplace=True)

prev_statelist = [] #memoizing the states
history = [0] * len(pos)
viterbi_final_taglist = []

#for i in range(0, len(dftest)): #every word
for i in range(0, len(dftest)):
    if dftest.word[i] in wordlist:
        curword = dftest.word[i]
        #print(curword)
    else:
        curword = '<unk>' 

    curidx=wordlist.index(curword) 

    #print(curidx)
    
    if(curword != '.'): #not end of sentence
    
        if(dfdev['index'][i] == 1):#first word of sentence
            #initial *  emission
            prev_prob = [0] * len(pos)
            prevstates = []

            for j in range(0, len(pos)): #was pos
                prev_prob[j]=initial[pos[j]] * emissionmat[j][curidx]
            prevstates = [-1] *len(pos) #wasnone
            prev_statelist.append(prevstates)
            history = prev_prob
            #print(prevstates)

        else: # subsequent word in sentence 
            # trans * emission * previous

            prev_prob = [0] * len(pos)
            prevstates = [-1] * len(pos) #was0

            for j in range(0, len(pos)): #was pos
                curr_max_prob = 0
                curr_state = -1

                for k in range(0,len(pos)): #was pos

                    prob = history[k] * transitionmat[k][j] * emissionmat[j][curidx]
                    #print(curword)
                    #print(curword,history[k],transitionmat[k][j],emissionmat[j][wordlist.index(curword)])
                    if prob > curr_max_prob:
                        curr_max_prob = prob
                        curr_state = k
                        #print(prob,k)
                prev_prob[j] = curr_max_prob
                prevstates[j] = curr_state

            history = prev_prob
            #print(prevstates)
            prev_statelist.append(prevstates)
        
    else: #end of sentence
        
#         prevstates[-1]* len(pos)
#         prevstates[10] = 1
#         prev_statelist.append(prevstates)
        
             
              
        viterbi_taglist = []
        viterbi_taglist.append(".")
        
        
        prev_idx = np.argmax(history)
        prev_tag = pos[prev_idx]
        viterbi_taglist.append(prev_tag)
        
#         l=len(prev_statelist)-2
        
#         while(prev_tag != None):
#             prev_idx = prev_statelist[l][prev_idx]
#             prev_tag = str(poslist[prev_idx])
#             viterbi_taglist.append(prev_tag)
#             l -=1
        for i in range(1,len(prev_statelist)):
            #print(i)
            prev_idx = prev_statelist[len(prev_statelist)-i][prev_idx]
            prev_tag = str(poslist[prev_idx])
            viterbi_taglist.append(prev_tag)

        #print(prev_statelist)
        viterbi_taglist.reverse()
        viterbi_final_taglist.append(viterbi_taglist)
            
        
        prev_statelist = [] #memoizing the states
        history = [0] * len(pos)
        
        #argax of history
        #assign index (argmax) = curr tag
        #pos[value of argmax] = prev tag






# In[ ]:





# In[22]:


## print value of viterbi hmm decoding
i = 0
j=1
with open("viterbi.out", 'w') as f:
    for i in range(0, len(dftest)):
        
      
        
        if(dftest['index'][i]==1 and i!=0):
            f.write('\n')
            j=1
            f.write(str(j) + '\t' + dftest.word[i] + '\t' + str(viterbi_ans[i]) + '\n')
            
            
        else:
            if(i!=0):
                j+=1
            f.write(str(j) + '\t' + dftest.word[i] + '\t' + str(viterbi_ans[i]) + '\n')
            
        i +=1
        

f.close()



# In[ ]:




