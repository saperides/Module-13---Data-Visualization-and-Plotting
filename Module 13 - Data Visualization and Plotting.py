#!/usr/bin/env python
# coding: utf-8

# Find a dataset you'd like to explore. This can be something you're familiar with or something new. Create a Jupyter notebook and then:
# 
# Choose one variable and plot that variable four different ways.
# 
# Choose two continuous variables, and plot them three different ways.
# 
# Choose one continuous variable and one categorical variable, and plot them six different ways.
# 
# Give the pros and cons of each plot you create. You can use variables from multiple datasets if you like.
# 
# Find a dataset with at least four continuous variables and one categorical variable. Create one master plot that gives insight into the variables and their interrelationships, including:
# 
# Probability distributions
# 
# Bivariate relationships
# 
# Whether the distributions or the relationships vary across groups
# 
# Accompany your plot with a written description of what you see.

# In[43]:


# import modules and read ebola csv as a dataframe
import os
os.chdir('/Users/sophiaperides/Desktop/Thinkful')
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

mwd = pd.read_csv('Minimum_Wage_Data.csv')
mwd_df = pd.DataFrame(mwd)
print('Minimum Wage Columns')
print(mwd_df.columns)
print('\n')
print('World Happiness Columns')
wh_2016 = pd.read_csv('world_happiness_2016.csv')
wh_2016 = pd.DataFrame(wh_2016)
print(wh_2016.columns)


# In[12]:


mass = mwd_df[mwd_df['State'] == 'Massachusetts' ]
mass_low = mass[['Year','Low.Value']]


# In[33]:


plt.hist(mass_low['Low.Value'], bins=20)
plt.title('Plot 1: Minimum Wage in Massachusetts')
plt.xlabel('Minimum Wage in USD')
plt.show()
print('This plot isn\'t particularly usefule. We can see that more often than not, \n the minimum wage was below $5, but it gives very little information.')

sns.relplot(x='Year', y='Low.Value', data=mass_low)
plt.title('Plot 2: Minimum Wage in Massachusetts')
plt.show()

sns.relplot(x='Year', y='Low.Value', data=mass_low, kind='line')
plt.title('Plot 3: Minimum Wage in Massachusetts')
plt.show()

print('Both the scatter and ilne plots are more useful. We can see a steady increase \n over time as well as points of stagnation.')

ax = sns.boxplot(y='Low.Value',data=mass_low)  
plt.title('Plot 4: Minimum Wage in Massachusetts')
sns.despine(offset=10, trim=True)
ax.set(xlabel='', ylabel='Minimum Wage')
plt.show()

print('I don\'t feel that the boxplot is particularly useful. Similiarly to the histogram, \n we can see that the minimum wage has at or below ~$4.')

sns.catplot(y='Low.Value', kind='boxen', data=mass_low)


# In[51]:


# Choose two continuous variables, and plot them three different ways.
hap_score = wh_2016[['Happiness Score']]
trust = wh_2016[['Trust (Government Corruption)']]
print(hap_score[:5])
print(trust[:5])


# In[89]:


sns.relplot(x='Happiness Score', y='Trust (Government Corruption)', data=wh_2016)
plt.title('Plot 1: Trust in Government Versus Happiness Scores')
plt.show()
print('I like this plot, I think it gives us a good idea of the correlation between how \n much trust a people has in its government and their perceived notion of happiness.')


sns.relplot(x='Happiness Score', y='Trust (Government Corruption)', data=wh_2016, kind='line')
plt.title('Plot 2: Trust in Government Versus Happiness Scores')
plt.show()
print('This plot has a lot going on. While we can see the correlation, it\'s too busy and \n simply not a good option for the data.')

sns.lmplot(x='Happiness Score', y='Trust (Government Corruption)', data=wh_2016, # Data.
               fit_reg=True, # The regression line also includes a 95% confidence envelope.
               scatter_kws={'alpha':0.4})
g.set_ylabels("Trust")
g.set_xlabels("Happiness Score")
plt.title('Scatterplot with regression line: Trust in Government Versus Happiness Scores')
plt.show()
print('Though this plot is similar to the first, it\'s a bit more informatative. We can \n see that thee is definitely a positive correlation between trust and happiness scores.')
print('I\'d definitely like to see other options (a histogram in particular) for plotting this type of data.')



# In[111]:


print(wh_2016['Region'].unique())


# In[131]:


# Updated the data with a 'Continent' column
wh_2016['Continent'] = []
def continent_column(df):
    for row in df:
        if df.loc[row, 'Region'] == 'Western Europe' | df['Region'] == 'Central and Eastern Europe':
            df.loc[row, 'Continent'] = 'Europe'
        elif df.loc[row, 'Region'] == 'Australia and New Zealand':
            df.loc[row, 'Continent'] = 'Australia'
        elif df.loc[row, 'Region'] == 'Middle East and Northern Africa' | df.loc[row, 'Region'] == 'Sub-Saharan Africa':
            df.loc[row, 'Continent'] = 'Africa'
        elif df.loc[row, 'Region'] == 'Southeastern Asia' | df.loc[row, 'Region'] == 'Eastern Asia' | df.loc[row, 'Region'] == 'Southern Asia':
            df.loc[row, 'Continent'] = 'Asia'
        else:
            df.loc[row,'Continent'] = df['Region']
    return df
continent_column(wh_2016)


# In[110]:


#Choose one continuous variable and one categorical variable, and plot them six different ways.
# Region and happiness score



scatter = sns.catplot(x='Region', y='Happiness Score', size=10, data=wh_2016)
violin = sns.catplot(x='Region', y='Happiness Score', kind='violin', data=wh_2016)
boxen = sns.catplot(x='Region', y='Happiness Score', kind='boxen', data=wh_2016)


# In[ ]:




