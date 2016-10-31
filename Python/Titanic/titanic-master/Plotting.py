# Import the necessary modules
import numpy as np
import pandas as pd
import rpy2.robjects as robj
import rpy2.robjects.pandas2ri # for dataframe conversion
from rpy2.robjects.packages import importr
 
# First, make some random data
x = np.random.normal(loc = 5, scale = 2, size = 10)
y = x + np.random.normal(loc = 0, scale = 2, size = 10)
 
# Make these into a pandas dataframe. I do this because
# more often than not, I read in a pandas dataframe, so this
# shows how to use a pandas dataframe to plot in ggplot
testData = pd.DataFrame( {'x':x, 'y':y} )
# it looks just like a dataframe from R


# print testData
 
# Next, you make an robject containing function that makes the plot.
# the language in the function is pure R, so it can be anything
# note that the R environment is blank to start, so ggplot2 has to be
# loaded

import pandas as pd
df  = pd.read_csv('/home/shubhanshu/ML Python/Titanic/titanic-master/train.csv')
age_mean = df['Age'].mean()
df['Age'] = df['Age'].fillna(age_mean)
from scipy.stats import mode
mode_embarked = mode(df['Embarked'])[0][0]
df['Embarked'] = df['Embarked'].fillna(mode_embarked)
df['Ages'] = df['Age'].map(lambda x: int(x/10)*10)
df['Fares'] = df['Fare'].map(lambda x: int(x/10)*10)


# plotFunc = robj.r("""
#  library(ggplot2)
 
# function(df){
#  p <- ggplot(df, aes(x, y)) +
#  geom_point( )
 
# print(p)
#  }
# """)
 

# # import graphics devices. This is necessary to shut the graph off
# # otherwise it just hangs and freezes python
# gr = importr('grDevices')
 
# # convert the testData to an R dataframe
# robj.pandas2ri.activate()
# testData_R = robj.conversion.py2ri(testData)
 
# # run the plot function on the dataframe
# plotFunc(testData_R)


gr = importr('grDevices')
 
# convert the testData to an R dataframe
robj.pandas2ri.activate()


plotFunc_0 = robj.r("""
 library(ggplot2)
 
function(df){
 p <- qplot(Age, data=df, fill=factor(Fares), geom="histogram", binwidth=5)
 
print(p)
 }
""")
# import graphics devices. This is necessary to shut the graph off
# otherwise it just hangs and freezes python

testData_0 = robj.conversion.py2ri(df)

plotFunc_0(testData_0)


raw_input()

plotFunc_1 = robj.r("""
 library(ggplot2)
 
function(df){
 p <- qplot(factor(Pclass), data=df, fill=factor(Pclass), geom="bar") + coord_flip()

 
print(p)
 }
""")
# import graphics devices. This is necessary to shut the graph off
# otherwise it just hangs and freezes python

testData_1 = robj.conversion.py2ri(df)

plotFunc_1(testData_1)

 
# ask for input. This requires you to press enter, otherwise the plot
# window closes immediately
raw_input()

# shut down the window using dev_off()
gr.dev_off()
 
# # you can even save the output once you like it
# plotFunc_2 = robj.r("""
#  library(ggplot2)
 
# function(df){
#  p <- ggplot(df, aes(x, y)) +
#  geom_point( ) +
#  theme(
#  panel.background = element_rect(fill = NA, color = 'black')
#  )
 
# ggsave('rpy2_magic.pdf', plot = p, width = 6.5, height = 5.5)
#  }
# """)
 
# plotFunc_2(testData_R)

