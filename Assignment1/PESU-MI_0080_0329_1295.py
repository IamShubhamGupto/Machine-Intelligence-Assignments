import numpy as np
import pandas as pd
import random

'''Calculate the entropy of the enitre dataset'''
#input:pandas_dataframe
#output:int/float/double/large
DEBUG = 0
def get_entropy_of_dataset(df):
  entropy = 0
  values_last = df.keys()[-1]
  unique_values_last = df[values_last].unique()      #Cleaning the values of the last attribute and getting only unique values.
  for value in unique_values_last:
    temp = df[values_last].value_counts()[value]/len(df[values_last])         #Counting the number of positives as well as negatives in each iteration and dividing with the total count.
    entropy+= -temp*np.log2(temp)
  if DEBUG:
    print(entropy)
  return entropy



'''Return entropy of the attribute provided as parameter'''
	#input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
	#output:int/float/double/large
def get_entropy_of_attribute(df,attribute):
  entropy_of_attribute = 0
  unique_attr = df[attribute].unique() #Variables
  values_last = df.keys()[-1] #CLass 
  #print(df.keys())                  # Gives the positives and negatives of the last attribute.     
  unique_values_last = df[values_last].unique() #Target_variables # Gives the positives and negatives once.
  for cat in unique_attr:
    entropy_of_cat = 0  
    for value in unique_values_last:
      #Counting the number of positives as well as negatives in each iteration and dividing with the total count.
      bottom = (len(df[attribute][df[attribute] == cat]) )
      top = len(df[attribute][df[attribute] == cat][df[values_last] == value])

      temp = top/(bottom + np.finfo(np.float32).eps)
      entropy_of_cat += -temp*np.log2(temp + np.finfo(np.float32).eps)
     
    ratio = bottom/len(df[values_last])
    #print("DIFFerence",len(df), len(values_last))
    entropy_of_attribute += -ratio*entropy_of_cat
  if DEBUG:  
    print(entropy_of_attribute)
  return abs(entropy_of_attribute)


'''Return Information Gain of the attribute provided as parameter'''
	#input:int/float/double/large,int/float/double/large
	#output:int/float/double/large
def get_information_gain(df,attribute):
  '''
  information_gain = 0
  entropy_of_attribute = 0
  unique_attr = df[attribute].unique()
  values_last = df.keys()[-1]                   # Gives the positives and negatives of the last attribute.     
  unique_values_last = df[values_last].unique()
	#entropy_vals = []
  #frac_vals = []
  avg_info_gain = 0
  
  return information_gain
  '''
  information_gain = 0
  information_gain = get_entropy_of_dataset(df) - get_entropy_of_attribute(df,attribute)
  return information_gain


''' Returns Attribute with highest info gain'''  
	#input: pandas_dataframe
	#output: ({dict},'str')     
def get_selected_attribute(df):
   
  information_gains={}
  #selected_column=''
  all_cols = df.keys()[:-1]
  max_gain = -999999
  max_col = ''
  #eta = 0.001
  for col in all_cols:
    information_gains[col] = get_information_gain(df,col)
    if information_gains[col] > max_gain:
      max_gain = information_gains[col]
      max_col = col
  '''
	Return a tuple with the first element as a dictionary which has IG of all columns 
	and the second element as a string with the name of the column selected

	example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
	'''
  if DEBUG:
    print(information_gains,max_col)
  return (information_gains,max_col)



'''
------- TEST CASES --------
How to run sample test cases ?

Simply run the file DT_SampleTestCase.py
Follow convention and do not change any file / function names

'''
