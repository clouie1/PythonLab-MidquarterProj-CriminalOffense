# -*- coding: utf-8 -*-
"""
Created on Sat May 13 18:19:20 2017

Econ 294A Python Lab Midterm Project
@author: Christina Louie
Date: May 13, 2017
"""
# ------- libraries -------
import pandas as pd
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt


# ------- Some Preliminary Analysis -----------
filelocation = r'C:\Users\ChristinaL\Documents\Econ 294A Python Lab\crime_ca_2013.csv'
crime_dataset = pd.read_csv(filelocation)
#print(crime_dataset)

# -- Property Crime -- 
#plot of property crimes against population
plt.plot(crime_dataset['population'],crime_dataset['property_crime'],'go')
plt.suptitle("Figure 1: Plot of Property Crimes on Population")
plt.xticks(rotation=45)
plt.xlabel('Population')
plt.ylabel('Property Crimes')
plt.show()


# Find the largest number of property crimes
sorted_property = crime_dataset.sort_values(['property_crime'], ascending=False)
print(sorted_property.head(n=10))

# Find the average number of property crimes/offenses 
propertyCrimeAvg = crime_dataset['property_crime'].mean()
print(propertyCrimeAvg)

# Mean is 1883.0844155844156

# -- Violent Crime --
#plot of violent crimes against population
plt.plot(crime_dataset['population'],crime_dataset['violent_crime'],'ro')
plt.suptitle("Figure 2: Plot of Violent Crimes on Population")
plt.xticks(rotation=45)
plt.xlabel('Population')
plt.ylabel('Violent Crimes')
plt.show()


# Find the largest number of violent crimes
sorted_violent = crime_dataset.sort_values(['violent_crime'], ascending=False)
print(sorted_violent.head(n=10))

# Find the average number of violent crimes/offenses 
violentCrimeAvg = crime_dataset['violent_crime'].mean()
print(violentCrimeAvg)

# Mean is 269.6926406926407

# -- Relationship between property crime and violent crime -- 
# plot
plt.plot(crime_dataset['violent_crime'],crime_dataset['property_crime'],'bo')
plt.suptitle("Figure 3: Plot of Property Crimes on Violent Crimes")
plt.xlabel('Violent Crimes')
plt.ylabel('Property Crimes')
plt.xticks(rotation=45)
plt.show()

# linear regression
result1 = sm.ols(formula="property_crime ~ violent_crime",data=crime_dataset).fit(cov_type='HC3')
print(result1.params)
print(result1.summary())

# --- subset of data ---
# Define large population to be greater than or equal to 60,000 people
largePop_df = crime_dataset.loc[crime_dataset['population']>=60000]

# linear regression 
result2 = sm.ols(formula="property_crime ~ violent_crime",data=largePop_df).fit(cov_type='HC3')
print(result2.params)
print(result2.summary())

# Define small population to be less than 60,000 people
smallPop_df = crime_dataset.loc[crime_dataset['population']<60000]

result3 = sm.ols(formula="property_crime ~ violent_crime",data=smallPop_df).fit(cov_type='HC3')
print(result3.params)
print(result3.summary())

# --- sum of each variable to get the total number of crimes (each type) ---
total_violent = sum(list(crime_dataset['violent_crime']))
print("Total Violent Crimes: %d"  %(total_violent))

total_murder = sum(list(crime_dataset['murder_manslaughter']))
print("Total Murder and Manslaughter: %d"  %(total_murder))

total_rape = sum(list(crime_dataset['rape']))
print("Total Rape: %d"  %(total_rape))

total_robbery = sum(list(crime_dataset['robbery']))
print("Total Robbery: %d"  %(total_robbery))

total_assault = sum(list(crime_dataset['aggravated_assault']))
print("Total Aggravated Assault: %d"  %(total_assault))

total_property = sum(list(crime_dataset['property_crime']))
print("Total Property Crimes: %d"  %(total_property))

total_burglary = sum(list(crime_dataset['burglary']))
print("Total Burglary: %d"  %(total_burglary))

total_theft = sum(list(crime_dataset['larceny_theft']))
print("Total Larceny Theft: %d"  %(total_theft))

total_mvTheft = sum(list(crime_dataset['motor_vehicle_theft']))
print("Total Motor Vehicle Theft: %d"  %(total_mvTheft))

total_arson = sum(list(crime_dataset['arson']))
print("Total Arson: %d"  %(total_arson))

# Output:
#Total Violent Crimes: 124598
#Total Murder and Manslaughter: 1400
#Total Rape: 6064
#Total Robbery: 48035
#Total Aggravated Assault: 69099
#Total Property Crimes: 869985
#Total Burglary: 190417
#Total Larceny Theft: 539803
#Total Motor Vehicle Theft: 139765
#Total Arson: 6203

# Some possible correlation...
# --- relationship between rape and murder and manslaughter ---
result1 = sm.ols(formula="murder_manslaughter ~ rape",data=crime_dataset).fit(cov_type='HC3')
print(result1.params)
print(result1.summary())

# --- relationship between arson and burglary ---
result1 = sm.ols(formula="arson ~ burglary",data=crime_dataset).fit(cov_type='HC3')
print(result1.params)
print(result1.summary())




