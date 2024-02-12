# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 21:37:30 2023

@author: rajendra


Problem Statement: - 
A film distribution company wants to target audience based on their likes and dislikes, you as a Chief Data Scientist Analyze the data and come up with different rules of movie list so that the business objective is achieved.

1.1.	What is the business objective?
  Increased Viewer Engagement,segment the audience based on their likes and dislikes regarding movies.
1.2.	Are there any constraints?
    Data Availability and Quality,privacy
1.3 Write about the benefits/impact of the solution - in what way does the business (client) benefit from the solution provided?
  Enhanced Viewer Experience,Improved Marketing and Promotions
"""


from mlxtend.frequent_patterns import apriori,association_rules


movies=[]

#It reads a dataset assumed to be in CSV format from the file path "C:/2-dataset/groceries.csv". The dataset contains grocery transactions.

with open("C:/2-dataset/my_movies.csv") as f:movies=f.read()
#splitting data into seperate transaction using seperator.
#The dataset is split into individual transactions based on the newline character ("\n").

movies=movies.split("\n")
#earlier groceries datastructure is in string format now it will change
#we will have to seperate out each item from each transaction
movies_list=[]

#Each transaction is further split by commas (",") to create a list of items purchased in each transaction.
for i in movies:
   movies_list.append(i.split(","))



all_movies_list=[i for item in movies for i in item]

#The code calculates the frequency of each item in the entire dataset using the Counter class from collections.
#Uses Counter from the collections module to count the occurrences of each item in all_groceries_list and stores the result in item_frequencies.
from collections import Counter
item_frequencies=Counter(all_movies_list)

#Items and their frequencies are sorted in ascending order.
item_frequencies=sorted(item_frequencies.items(),key=lambda x:x[1])

#separates the items and frequencies into two separate lists, reversing the order to have the most frequent items first.
items=list(reversed([i[0] for i in item_frequencies]))

frequencies=list(reversed([i[1] for i in item_frequencies]))

#A bar plot is created using Matplotlib to visualize the top 10 most frequent items in the dataset.

import matplotlib.pyplot as plt
plt.bar(height=frequencies[0:4],x=list(range(0,4)))
plt.xlabel("items")
plt.ylabel("count")
plt.show()


#Creates a Pandas DataFrame groceries_series from the list groceries_list.
#Limits the DataFrame to the first 9835 rows (transactions).
#Renames the column in the DataFrame as "Transactions".
#Concatenates the items in each transaction with a separator "*" and converts it into a string.
#Converts this string data into a one-hot encoded DataFrame using get_dummies function
import pandas as pd

movies_series=pd.DataFrame(pd.Series(movies_list))
# the last row of the dataframe is empty so we will remove it
movies_series=movies_series.iloc[0:11,:]
movies_series

movies_series.columns=["Transactions"]
x=movies_series["Transactions"].str.join(sep="*")

#The transactions are converted into a one-hot encoded format using Pandas' get_dummies function to prepare the data for association rule mining.
x=x.str.get_dummies(sep="*")
frequent_itemsets=apriori(x,min_support=0.0075,max_len=4,use_colnames=True)


frequent_itemsets.sort_values("support",ascending=False,inplace=True)




#Association rules are generated from the frequent item sets using the association_rules function from mlxtend.
#The rules are sorted based on the "lift" metric in descending order.

rules=association_rules(frequent_itemsets,metric="lift",min_threshold=1)

#Displays the top 20 association rules and the top 10 rules with the highest "lift" metric.

rules.head(20)
rules.sort_values("lift",ascending=False).head(10)







