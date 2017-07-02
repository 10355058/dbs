
#Title:R script for Model Build and Testing
#Author: 
#: 

#Remarks I have tried include the build steps in sequence
#This covers Data Import,Cleaning, Model Build, Model Testing and variios Graphings
#Additional workings and various ad hoc test, and unused tests are included below the main sections
#These may help in understanding the process undertaken

library(ggplot2)
library(corrplot)
library(plotly)
library(sm)
packageVersion('plotly')

par(mfrow=c(1,1)) # one plot on one page



################################################################## Quick data munge  #################################
#Quick way to get the data munged if you want to jump to the Models

diamonds<-NULL

#Read from the .csv file
diamonds<-read.csv("diamonds.csv",header = T,stringsAsFactors = T, na.strings=c(NA,"NA"," NA"))


################################ Drop index Column X (row number) 

# Drop this column
diamonds$X<-NULL


################################ Drop zero dimesions 

diamonds<-subset(diamonds,diamonds$x!=0)
diamonds<-subset(diamonds,diamonds$y!=0)
diamonds<-subset(diamonds,diamonds$z!=0)

################################ # Reindex the dataframe 
old_rownames<-row.names(diamonds)
rownames(diamonds) <- seq(length=nrow(diamonds)) 

#Drop these outliers from data 
diamonds<-diamonds[(diamonds$y<58.9),]
diamonds<-diamonds[(diamonds$z<31.8),]

#Well we have out Addition high y value 

diamonds<-diamonds[(diamonds$y!=31.8),]

# Reindex the dataframe
old_rownames<-row.names(diamonds)
rownames(diamonds) <- seq(length=nrow(diamonds)) 


#So we drop table value as being likely to be unrepresentative

diamonds<-diamonds[(diamonds$table!=95),]

# Reindex the dataframe
old_rownames<-row.names(diamonds)
rownames(diamonds) <- seq(length=nrow(diamonds)) 


# create a new column lprice
diamonds$lprice<-log(diamonds$price)

#Add our additional flag


diamonds$ideal_flag <- ifelse(diamonds$cut =='Ideal', "True", "False")

################################################################## Data Import  #################################
setwd()
getwd()
diamonds<-NULL

#Read from the .csv file
diamonds<-read.csv("diamonds.csv",header = T,stringsAsFactors = T, na.strings=c(NA,"NA"," NA"))

# If needed could remove instances which have at least one NA variable - does nothing as no missing values
diamonds <- diamonds[complete.cases(diamonds), ]




################################################################## Initial examination of Structure of diamonds data  #################################

str(diamonds)


#Summary statistics for diamonds data
summary(diamonds)

#points of interest


#price price in US dollars (\$326--\$18,823)

#carat weight of the diamond (0.2--5.01)

#cut quality of the cut (Fair, Good, Very Good, Premium, Ideal)

#color diamond colour, from J (worst) to D (best)

#clarity a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))

#x length in mm (0--10.74)

#y width in mm (0--58.9)

#z depth in mm (0--31.8)

#depth total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)

#table width of top of diamond relative to widest point (43--95)


# Already we can see that the x,y,z values have mins of zero - Does this really make sense?



#carat - weight of diamond
#cut - shape diamond is cut
#color - white, blue etc
#clarity - inclusions etc
#depth has value ranging for 0 to :99999999
#table sees redundant
#price - What is sells for
#x dimension
#y dimension
#z dimension

names(diamonds)

# Column X is a row number
# Drop this column
diamonds$X<-NULL

names(diamonds)

#Some basic plotting just to confirm what we think we know from looking at summary

summary(diamonds$price)

hist(diamonds$price)
plot(diamonds$cut)
hist(diamonds$carat)
hist(diamonds$depth)
hist(diamonds$table)

hist(diamonds$x)
hist(diamonds$y)
hist(diamonds$z)


str(diamonds)

summary(diamonds)

dim(diamonds) # Just so we'ew clear on what out dataset likes like

#NOw lest look at these x,y,z values


dim(diamonds[diamonds$x==0,])

#Lets just take a look at these 8 rows
print (diamonds[diamonds$x==0,])

#We can do the same with y
dim(diamonds[diamonds$y==0,])

#Lets just take a look at these 7 rows
print (diamonds[diamonds$y==0,])

dim(diamonds[diamonds$z==0,])
#Lets just take a look at these 20 rows
print (diamonds[diamonds$z==0,])

                                               
#It looks like we have some missing (default=0) or misrecorded values but can we really be sure?
#Given the small number of cases, the knock effect on the calculated table value and the likely effect 
#on any pricing model incorporaing such unusually shaped diammonds its probably advisable to drop them

#So we'll loose 20 rows from 53940 - that shouldn't be too problematic

################################ Drop zero dimesions #################################

diamonds<-subset(diamonds,diamonds$x!=0)
diamonds<-subset(diamonds,diamonds$y!=0)
diamonds<-subset(diamonds,diamonds$z!=0)



################################ # Reindex the dataframe ################################ 
old_rownames<-row.names(diamonds)
rownames(diamonds) <- seq(length=nrow(diamonds)) 

#Make sure they're gone
dim(diamonds[diamonds$price==0,])


summary(diamonds)
dim(diamonds) 

################################ # Check for missing value ################################ 
sum(is.na(diamonds$price)) #None

sum(is.na(diamonds$carat)) #None

sum(is.na(diamonds$depth)) #None

################################ After the zero values are dropped ################################ 

####Lets see what that does to the Distribution of the data


#Quick Hstogram plotting test - just to make sure we can plot
diamonds_temp<-c(1,1,1,2,3,3,4,4,4)
plot_ly(x = ~diamonds_temp, type = "histogram")
hist(diamonds_temp)



# try a Histogram
hist(diamonds$price)

summary(diamonds$price)

#Maybe try a little fancier plot
plot_ly(x = ~diamonds$price, type = "histogram")

#So it seems we have more cheap stones that expensive
#We can conform this with out quantile at varies values - 50% of diamonds priced less than ~$2405
quantile(diamonds$price,c(0.50,0.75,0.90,0.95,0.99))


#So just what is the minimum price for a diamond

min(diamonds$price)
diamonds_temp<-  table(diamonds$price)
diamonds_temp[names(diamonds_temp)==326]

# or another way to do the same thing
diamonds$price[diamonds$price==326]

#Lets just try a density plot
plot(density(diamonds$price,na.rm=TRUE))

par(mfrow=c(2,1)) # multiple plots

plot_ly(x = ~diamonds$price, type = "histogram")
plot_ly(x = ~diamonds$price, type = "density")

# can we see anything interesting with x,y,z if we plot them as histograms on same axis
plot_ly(alpha = 0.6) %>%
  add_histogram(x = ~diamonds$x, name = 'x') %>%
  add_histogram(x = ~diamonds$y, name = 'y') %>%
  add_histogram(x = ~diamonds$z, name = 'z') %>%
  layout(barmode = "overlay") 

#Well the plots overlap but why the hugh area off to the right?

max(diamonds$x)
max(diamonds$y)
max(diamonds$z)

# Better take a look at the x,y,z quantiles
quantile(diamonds$x,c(0.50,0.75,0.90,0.95,0.99,1.0))
quantile(diamonds$y,c(0.50,0.75,0.90,0.95,0.99,1.0))
quantile(diamonds$z,c(0.50,0.75,0.90,0.95,0.99,1.0))


#So we have outliers for y, and z 


par(mfrow=c(1,1)) # one plot on one page

ggplot(diamonds, aes(x=diamonds$price)) +
  geom_histogram(aes(y = ..density..), binwidth=density(diamonds$price)$bw) + geom_density(fill="yellow", alpha = 0.2)


ggpplot(diamonds,aes(x=diamonds$price))+geom_histogram()
ggplotly()



# Histogram of differnt types
ggplot(diamonds, aes(x=price)) + 
  geom_histogram( # Histogram with density instead of count on y-axis
                 binwidth=.5,
                 colour="black", fill="white")

# Histogram overlaid with kernel density curve
ggplot(diamonds, aes(x=price)) + 
  geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                 binwidth=.5,
                 colour="black", fill="white") +
  geom_density(alpha=.2, fill="#FF6666")  # Overlay with transparent density plot


################################ Deal with the y, z Outliers  ################################ 

#So we have a few outliers well beyond the majority of values for y,z
#Again a small number but their effect is likely to be large if they are used in a pricing model
#So we drop them

par(mfrow=c(1,1)) # two plots on one page

#Just remind outselves of where we are
hist(diamonds$x)
hist(diamonds$y)
hist(diamonds$z)

summary(diamonds$x)
summary(diamonds$y)
summary(diamonds$z)

max(diamonds$y)
max(diamonds$z)


quantile(diamonds$y,c(0.50,0.75,0.90,0.95,0.99,1.0))
quantile(diamonds$z,c(0.50,0.75,0.90,0.95,0.99,1.0))
# A scatter plot for y vs price

plot_ly(data = diamonds, type="scatter",mode= "markers",x = ~y, y = ~price)

#So that quite clear we have 2 point far to the right of the rest of the data

# A scatter plot for z vs price - we've got a single data point out to the right

plot_ly(data = diamonds, type="scatter",mode= "markers",x = ~z, y = ~price)

################################ #Get the data with the outlier values for y and z ################################ 

diamonds[(diamonds$y==58.9),]

diamonds[(diamonds$z==31.8),]

################################  #Check we have the right rows ################################ 

diamonds[(24059),]
diamonds[(48394),]

#Just in case we get make any mistakes we'll backup out current dataframe
data<-diamonds

#diamonds<-data

summary(data)

################################  Drop these problematic outliers from y, z ################################ 


#Drop these outliers from data 
diamonds<-diamonds[(diamonds$y<58.9),]
diamonds<-diamonds[(diamonds$z<31.8),]

summary(diamonds)


par(mfrow=c(1,1)) # multiple plots
#Ok, z looks good but we have one more to deal with in y
plot_ly(data = diamonds, type="scatter",mode= "markers",x = ~diamonds$z, y = ~diamonds$price)
plot_ly(data = diamonds, type="scatter",mode= "markers",x = ~diamonds$y, y = ~diamonds$price)



#Well we have out Addition high y value 

diamonds[(diamonds$y==31.8),]


summary(diamonds$y)
quantile(diamonds$y,c(.90,.99,.999,0.9999))

# Reindex the dataframe
old_rownames<-row.names(diamonds)
rownames(diamonds) <- seq(length=nrow(diamonds)) 

# Let just check one last time
summary(diamonds)

str(diamonds)

quantile(diamonds$x,c(.90,.99,.999,0.9999,1.0))

quantile(diamonds$y,c(.90,.99,.999,0.9999,1.0))

quantile(diamonds$z,c(.90,.99,.999,0.9999,1.0))



summary(diamonds)

#So we drop this value as being likely to be unrepresentative

#diamonds<-diamonds[(diamonds$z!=31.8),]
#diamonds<-diamonds[(diamonds$y!=58.9),]

diamonds<-diamonds[(diamonds$y!=31.8),]

# Reindex the dataframe
old_rownames<-row.names(diamonds)
rownames(diamonds) <- seq(length=nrow(diamonds)) 


#Just remind outselves of where we are
hist(diamonds$x)
hist(diamonds$y)
hist(diamonds$z)

#And Try that Overlay plot again

plot_ly(alpha = 0.6) %>%
  add_histogram(x = ~diamonds$x, name = 'x') %>%
  add_histogram(x = ~diamonds$y, name = 'y') %>%
  add_histogram(x = ~diamonds$z, name = 'z') %>%
  layout(barmode = "overlay") 


################################  What about the other variables ################################ 

#Well x,y,and z are used to calculate depth

#Let's just check the depth values - just in case

summary(diamonds$depth)

quantile(diamonds$depth,c(.90,.99,.999,0.9999,1.0))

hist(diamonds$depth)

#Well that looks ok - no obvious issue there


################################  What about table? ################################ 

# The other field is table - a 
summary(diamonds$table)

#The max might be a woth a look

quantile(diamonds$table,c(.90,.99,.999,0.9999,1.0))


hist(diamonds$table)

# Maybe this is clearer
ggplot(diamonds, aes(x=diamonds$table)) +
  geom_histogram(aes(y = ..density..), binwidth=density(diamonds$table)$bw) + geom_density(fill="yellow", alpha = 0.2)



# Histogram overlaid with kernel density curve
ggplot(diamonds, aes(x=table)) + 
  geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
                 binwidth=.5,
                 colour="black", fill="white") +
  geom_density(alpha=.2, fill="#FF6666")  # Overlay with transparent density plot




# Again a scatter plot might actually be the best thing to show this

plot_ly(data = diamonds, type="scatter",mode= "markers",x = ~diamonds$table, y = ~diamonds$price)


quantile(diamonds$table,c(.90,.99,.999,0.9999,1.0))


################################ #Get the data with the outlier value for table ################################ 

# Reindex the dataframe
old_rownames<-row.names(diamonds)
rownames(diamonds) <- seq(length=nrow(diamonds)) 


diamonds[(diamonds$table==95),]



################################  #Check we have the right row ################################ 


diamonds[(24921),]


################################  Drop table outlier ################################ 


#So we drop this value as being likely to be unrepresentative

diamonds<-diamonds[(diamonds$table!=95),]


#Check out data again


plot_ly(data = diamonds, type="scatter",mode= "markers",x = ~diamonds$table, y = ~diamonds$price)





################################  What about carat? ################################ 


#Here are the carat weights - nothing too strange
hist(diamonds$carat)
summary(diamonds$carat)

#So there are 12 diamonds at the minimum value
dim(diamonds[diamonds$carat==0.20,])

#And one at the max
dim(diamonds[diamonds$carat==5.01,])

#Lets get a better histogram if we can
plot_ly(x = ~diamonds$carat, type = "histogram")

# And a density ploy show a number of peaks - so  some type of weight bin?
plot(density(diamonds$carat,na.rm=TRUE))

plot(diamonds$carat)

#
hist(diamonds$carat)
plot(density(diamonds$carat,na.rm=TRUE))

#Does this look multimodal????

#Lets plot the Carat Weight gainst price
plot(diamonds$carat,diamonds$price)


#sm.density.compare(diamonds$price, diamonds$carat, xlab="Price by Carat")
#title(main="Price Distribution by Diamond Carat weight")

#So a Density Plot of Price by Carat
ggplot(diamonds, aes(diamonds$price, fill = diamonds$carat)) + 
  geom_density(alpha = 0.5, position = "stack") + 
  ggtitle("Stacked density chart")


################################  What about Colour? ################################ 


# Lets look at Colour

sm.density.compare(diamonds$price, diamonds$color, xlab="Price by Colour")
title(main="Price Distribution by Diamond Colour")


# Try this but not really an improvement
# create value labels
color.f <- factor(diamonds$color, levels= c('Fair','Good','Very Good','Premium','Ideal'),
                labels = c("Fair", "Good", "Very Good","Premium","Ideal"))

# plot densities
sm.density.compare(diamonds$price, diamonds$color, xlab="Price by Cut")
title(main="Price Distribution by Cuts")

# add legend via mouse click
colfill<-c(2:(2+length(levels(color.f))))
legend(locator(1), levels(color.f), fill=colfill) 




#So a nice Stacked Chart for the Density of Price by Colour
ggplot(diamonds, aes(diamonds$price, fill = diamonds$color)) + 
  geom_density(alpha = 0.5, position = "stack") + 
  ggtitle("Stacked density chart")

#gain more low price tahn high price diamonds, but a peak around $5000

#A kernel density plot
ggplot(diamonds, aes(x = diamonds$price)) + 
  geom_density(aes(fill = diamonds$color), alpha = 0.5) + 
  ggtitle("Kernel Density estimates by Color")

# A box plot might be more suitable
plot_ly(diamonds, x = ~diamonds$price, color = ~diamonds$color, type = "box")


head(diamonds)
head(data)

summary(diamonds$color)

#Try a Vertical box plot - and we naturally get out Order right
plot_ly(diamonds, y = ~diamonds$price, color = ~diamonds$color, type = "box")



################################  What about Cut? ################################ 

# So here are the cut levels we expected
plot(diamonds$cut)

# A box plot might be more suitable
plot_ly(diamonds, x = ~diamonds$price, color = ~diamonds$cut, type = "box")

# A Vertical plot still does not order as we would like
plot_ly(diamonds, y = ~diamonds$price, color = ~diamonds$cut, type = "box")


plot_ly(ggplot2::diamonds, y = ~price, color = ~cut, type = "box")



boxplot(diamonds$price[diamonds$cut=='Fair'],diamonds$price[diamonds$cut=='Good'],
        diamonds$price[diamonds$cut=='Very Good'],diamonds$price[diamonds$cut=='Premium'],diamonds$price[diamonds$cut=='Ideal']
        ,names=c("Fair","Good","Very Good","Premium","Ideal"),ylab="Diamond price US$")

plot_ly(ggplot2::diamonds, y = ~price, color = ~cut, type = "box")


#Lets reorder the cuts
boxplot(diamonds$price[diamonds$cut=='Fair'],diamonds$price[diamonds$cut=='Good'],
        diamonds$price[diamonds$cut=='Very Good'],diamonds$price[diamonds$cut=='Premium'],diamonds$price[diamonds$cut=='Ideal']
        ,names=c("Fair","Good","Very Good","Premium","Ideal"),ylab="Diamond price US$")



#cut.f <- factor(diamonds$color, levels= c('Fair','Good','Very Good','Premium','Ideal'),
                  labels = c("Fair", "Good", "Very Good","Premium","Ideal"))
#levels(factor(diamonds$color))
#cut.f <- levels(factor(diamonds$color))







#Just how do the means for the predicting factor cut differ or do they?
diamonds_temp = aov(price ~ cut, data=diamonds)

summary(diamonds_temp)

#Studying the output of the ANOVA table above we see that the F-  statistic is 
#174.3 with a p-  value equal to <2e-16
#We clearly reject the null hypothesis of equal means for all cut groups. 

#Lets use Tukey to see just how the means actually differ
TukeyHSD(diamonds_temp, conf.level =  0.95)


#Tukey multiple comparisons of means
#95% family-wise confidence level

#Fit: aov(formula = price ~ cut, data = diamonds)

#$cut
#                      diff         lwr        upr     p adj
#Good-Fair         -425.48206  -736.04198 -114.92214 0.0017428
#Ideal-Fair        -894.88023 -1174.24431 -615.51616 0.0000000
#Premium-Fair       227.24521   -57.53493  512.02534 0.1885094
#Very Good-Fair    -370.05494  -656.92029  -83.18959 0.0039654
#Ideal-Good        -469.39817  -640.40331 -298.39304 0.0000000
#Premium-Good       652.72727   473.01028  832.44426 0.0000000
#Very Good-Good      55.42712  -127.57616  238.43040 0.9226252
#Premium-Ideal     1122.12544  1004.24831 1240.00257 0.0000000
#Very Good-Ideal    524.82529   401.99605  647.65453 0.0000000
#Very Good-Premium -597.30015  -731.99411 -462.60619 0.0000000



plot(diamonds_temp)

(TukeyHSD(diamonds_temp, conf.level =  0.95))

(summary(TukeyHSD(diamonds_temp, conf.level =  0.95)))

sum_test = str(summary(TukeyHSD(diamonds_temp, conf.level =  0.95)))
sum_test
names(sum_test)

#We can try the same for color

# plot color

plot(density(diamonds$price,na.rm=TRUE))

plot(diamonds$color)

library(sm)

sm.density.compare(diamonds$price, diamonds$cut, xlab="Price Per Cut")
title(main="Price Distribution by Diamond Cut")


diamonds_temp = aov(price ~ color, data=diamonds)

summary(diamonds_temp)
#summary(diamonds_temp)
#               Df    Sum Sq   Mean Sq F value Pr(>F)    
#color           6 2.677e+10 4.462e+09   289.7 <2e-16 ***
#Residuals   53909 8.303e+11 1.540e+07                   
#---
#  Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1


TukeyHSD(diamonds_temp, conf.level =  0.95)

# TukeyHSD(diamonds_temp, conf.level =  0.95)
#Tukey multiple comparisons of means
#95% family-wise confidence level

#Fit: aov(formula = price ~ color, data = diamonds)

#$color
#      diff        lwr        upr     p adj
#E-D  -91.1387 -273.98301   91.70561 0.7630382
#F-D  555.9607  372.10801  739.81333 0.0000000
#G-D  828.7701  650.92628 1006.61395 0.0000000
#H-D 1311.9042 1122.43175 1501.37659 0.0000000
#I-D 1921.8676 1711.01119 2132.72392 0.0000000
#J-D 2155.7116 1896.01571 2415.40746 0.0000000
#F-E  647.0994  480.64761  813.55113 0.0000000
#G-E  919.9088  760.11890 1079.69873 0.0000000
#H-E 1403.0429 1230.40397 1575.68176 0.0000000
#I-E 2013.0063 1817.13692 2208.87558 0.0000000
#J-E 2246.8503 1999.16836 2494.53221 0.0000000
#G-F  272.8094  111.86667  433.75222 0.0000120
#H-F  755.9435  582.23700  929.65000 0.0000000
#I-F 1365.9069 1169.09592 1562.71785 0.0000000
#J-F 1599.7509 1351.32367 1848.17816 0.0000000
#H-G  483.1341  315.80033  650.46778 0.0000000
#I-G 1093.0974  901.88765 1284.30723 0.0000000
#J-G 1326.9415 1082.92769 1570.95524 0.0000000
#I-H  609.9634  407.89272  812.03405 0.0000000
#J-H  843.8074  591.19290 1096.42193 0.0000000
#J-I  233.8440  -35.18144  502.86950 0.1375528


#Quick Plot for table
hist(diamonds$depth)
plot(density(diamonds$depth,na.rm=TRUE))

hist(diamonds$table)
plot(density(diamonds$table,na.rm=TRUE))



### 

hist(log(diamonds$price))
plot(density(log(diamonds$price),na.rm=TRUE))

hist(log(diamonds$price))


boxplot(diamonds$price[diamonds$cut=='Fair'],diamonds$price[diamonds$cut=='Good'],diamonds$price[diamonds$cut=='Very Good'],diamonds$price[diamonds$cut='Premium'],diamonds$price[diamonds$cut='Ideal'],
          names=c("Fair","Good","Very Good","Premium","Ideal"), 
          ylab="Diamond price US$", title("diamond price by Cut",cex=0.8)) #

boxplot(diamonds$price~diamonds$cut,ylab="Diamond price US$",title("diamond price by Cut"))

#Lets reorder the cuts
boxplot(diamonds$price[diamonds$cut=='Fair'],diamonds$price[diamonds$cut=='Good'],
         diamonds$price[diamonds$cut=='Very Good'],diamonds$price[diamonds$cut=='Premium'],diamonds$price[diamonds$cut=='Ideal']
         ,names=c("Fair","Good","Very Good","Premium","Ideal"),ylab="Diamond price US$")

#, title("diamond price by Cut",cex=0.8)


#Could do this but maybe not
# I reorder the groups order : I change the order of the factor data$names
#data$names=factor(data$names, levels=levels(data$names)[c(1,4,3,2)])

#The plot is now ordered !
#boxplot(data$value ~ data$names, col=rgb(0.3,0.5,0.4,0.6), ylab="value", xlab="names in desired order")


                
boxplot(diamonds$price~ diamonds$cut,las = 2)


boxplot(diamonds$price~ diamonds$color,las = 2)

boxplot(diamonds$price~ diamonds$clarity)


plot_ly(ggplot2::diamonds, y = ~price, color = ~cut, type = "box")

plot_ly(ggplot2::diamonds, y = ~price, color = ~cut, type = "area")

plot_ly(ggplot2::diamonds, x = ~cut,y = ~price, color = ~cut, type = "box")

plot_ly(ggplot2::diamonds, x = ~clarity,y = ~price, color = ~clarity, type = "box")

summary(diamonds$price[diamonds$cut=="Premium"])

min(diamonds$price)
tail(diamonds$price[diamonds$cut=="Premium"])

summary(diamonds$price)
quantile(diamonds$price)
mean(diamonds$price)
median(diamonds$price)  

quantile(diamonds$price[diamonds$cut=="Ideal"])
mean(diamonds$price[diamonds$cut=="Ideal"])
median(diamonds$price[diamonds$cut=="Ideal"])   
     
     
quantile(diamonds$price[diamonds$cut=="Premium"])


hist(diamonds$price[diamonds$cut=='Ideal'])

hist(diamonds$price[diamonds$cut=='Fair'])







#levels(diamonds$cut)[c(1,2,5,4,3)]

plot_ly(ggplot2::diamonds, x = ~cut, y = ~price, color = ~clarity, type = "box") %>% layout(boxmode = "group")


plot_ly(ggplot2::diamonds, x = (diamonds$cut)[c(1,2,5,4,3)], y = ~price, color = ~clarity, type = "box") %>% layout(boxmode = "group")



boxplot(log(diamonds$price)~diamonds$cut,las = 2)

plot(diamonds$price,diamonds$cut)


#Check Normal distribution of Fields
qqnorm(diamonds$price)


cor(diamonds$carat,diamonds$price)


plot(diamonds)
plot(diamonds$carat)

summary(diamonds$price)

hist(diamonds$price)
hist(diamonds$carat)
hist(diamonds$depth)
hist(diamonds$table)

#log transform of price and carat to mke more normal but can't use with zero values or neg

#log(x+1) is data right skewed to avaoid




#depth percentage = z / mean(x, y) = 2 * z / (x + y)

head(2 * diamonds$z / (diamonds$x + diamonds$y))
head(diamonds)
########################################## Correlation #######################

####
#library(corrplot)
#corrplot(cor(diamonds[, -1]))

#library(corrgram)
#corrgram(diamonds, order=NULL, lower.panel=panel.shade,
         #upper.panel=NULL, text.panel=panel.txt,
         #main="Diamond data")

###


#plot(diamonds)

unique(diamonds$price)
head(diamonds)

#Count of price per count
aggregate(price ~ cut, diamonds, FUN = function(x) length(unique(x)))


aggregate(carat ~ cut, diamonds, FUN = function(x) length(unique(x)))



head(diamonds)

cor(diamonds[is.numeric(diamonds)])
diamonds_numeric<- sapply(diamonds, is.numeric)


corrplot(diamonds[ , diamonds_numeric])


######################################### Correlations Matrix for Dataset
diamonds_matrix <- cor(diamonds[sapply(diamonds, is.numeric)]) # get correlations

diamonds_temp<-cor(diamonds[sapply(diamonds, is.numeric)])
diamonds_temp<-diamonds_temp[which(diamonds_temp>0.50)]

diamonds_temp<-ifelse((diamonds_temp>0.5),diamonds_temp,0)
diamonds_temp

diamonds_temp<-ifelse((diamonds_temp>0.5),1,0)

diamonds_temp




library('corrplot') #package corrplot

diamonds_matrix <- cor(diamonds[sapply(diamonds, is.numeric)]) # get correlations
corrplot(diamonds_matrix, method = "circle") #plot matrix
corrplot.mixed(diamonds_matrix) #plot matrix
corrplot(diamonds_matrix, method="number")
cor(diamonds[sapply(diamonds, is.numeric)])


library('corrplot') #package corrplot
diamonds_matrix <- cor(diamonds[sapply(diamonds, is.numeric)]) # get correlations



corrplot(diamonds_matrix, method = "circle") #plot matrix

corrplot(diamonds_matrix, method = "circle") #plot matrix
corrplot.mixed(diamonds_matrix) #plot matrix
corrplot(diamonds_matrix, method="number")

#diamonds_matrix[c("price")]

View(diamonds_matrix[,"price"])

#M[87,]
#M[87, ]> 0.3 



#M[M[87,] > 0.3, ]


#subset(M, M[87, ] < 0)
       
#str(as.data.frame(M)) 
       
#str(as.data.frame(M)) 
       

################################################## Linear Models ##################


# The Linear Model#carat and size dimensions # R2 0.8588  - we could expect this given the correlations

fit1<-lm(diamonds$price~ carat+x+y+z,data=diamonds)
summary(fit1)
plot(fit1)

#> summary(fit1)

#Call:
  #lm(formula = diamonds$price ~ carat + x + y + z, data = diamonds)

#Residuals:
  #Min       1Q   Median       3Q      Max 
#-23550.6   -622.2    -46.9    348.0  12911.5 

#Coefficients:
  #Estimate Std. Error t value Pr(>|t|)    
#(Intercept)  2875.16     111.99   25.67   <2e-16 ***
  #carat       10904.91      67.39  161.82   <2e-16 ***
  #x           -3390.97     115.72  -29.30   <2e-16 ***
  #y            3622.18     113.24   31.99   <2e-16 ***
  #z           -2535.68      73.31  -34.59   <2e-16 ***
  #---
  #Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

#Residual standard error: 1498 on 53911 degrees of freedom
#Multiple R-squared:  0.8588,	Adjusted R-squared:  0.8588 
#F-statistic: 8.198e+04 on 4 and 53911 DF,  p-value: < 2.2e-16


fit_interaction<-lm(diamonds$price~ carat+x*y*z,data=diamonds)
summary(fit_interaction)


# Try the 4 C's - carat+cut+color+clarity
fit2<-lm(diamonds$price~ carat+cut+color+clarity,data=diamonds)
summary(fit2)
# > summary(fit2)
# 
# Call:
#   lm(formula = diamonds$price ~ carat + cut + color + clarity, 
#      data = diamonds)
# 
# Residuals:
#   Min       1Q   Median       3Q      Max 
# -16811.3   -680.2   -197.0    466.5  10396.6 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)  -7362.17      51.73 -142.32   <2e-16 ***
#   carat         8885.29      12.04  738.15   <2e-16 ***
#   cutGood        655.86      33.64   19.49   <2e-16 ***
#   cutIdeal       998.03      30.66   32.55   <2e-16 ***
#   cutPremium     868.40      30.94   28.07   <2e-16 ***
#   cutVery Good   848.46      31.28   27.12   <2e-16 ***
#   colorE        -210.55      18.31  -11.50   <2e-16 ***
#   colorF        -302.00      18.51  -16.32   <2e-16 ***
#   colorG        -504.27      18.12  -27.83   <2e-16 ***
#   colorH        -978.21      19.27  -50.77   <2e-16 ***
#   colorI       -1438.89      21.64  -66.49   <2e-16 ***
#   colorJ       -2323.82      26.71  -86.99   <2e-16 ***
#   clarityIF     5418.40      52.19  103.81   <2e-16 ***
#   claritySI1    3573.06      44.67   79.99   <2e-16 ***
#   claritySI2    2626.18      44.86   58.54   <2e-16 ***
#   clarityVS1    4534.16      45.61   99.42   <2e-16 ***
#   clarityVS2    4216.92      44.91   93.89   <2e-16 ***
#   clarityVVS1   5069.16      48.28  105.00   <2e-16 ***
#   clarityVVS2   4966.19      46.96  105.76   <2e-16 ***
#   ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# Residual standard error: 1156 on 53897 degrees of freedom
# Multiple R-squared:  0.9159,	Adjusted R-squared:  0.9159 
# F-statistic: 3.261e+04 on 18 and 53897 DF,  p-value: < 2.2e-16

#Well no surprses there

plot(fit2)




# The Linear Model#carat and size dimensions + cut #R2 0.8635
fit3<-lm(diamonds$price~ carat+x+y+z+cut,data=diamonds)
summary(fit3)




#Lets see if the model differ
anova(fit1,fit2)


#Linear Model usng all predictors #Adjusted R-squared:  0.9205 
fit_full<-lm(diamonds$price~. ,data=diamonds) 
summary(fit_full)

plot(fit_full)


which(fit$residuals < -2)

(diamonds[which(fit_full$residuals < -2),])



plot(diamonds,fit_full$residuals)

rownames(diamonds) <- seq(length=nrow(diamonds)) 

rownames(diamonds)

#We could Look at some outliers but there's one thing were forgetting in all the excitement



#Howver edspite the high R2 values we do kow that the price data is not normally distributed
#So we need to do a bit of work on price
hist(diamonds$price)

########################################################## Log Price #####################################################
# create a new column lprice
diamonds$lprice<-log(diamonds$price)
head(diamonds)

#Now that is more like a normal disribution
hist(lprice)

########################################################## Log Price Regression #####################################################

fit_log<-lm(diamonds$lprice~. ,data=diamonds)
summary(fit_log)

# > summary(fit_log)
# 
# Call:
#   lm(formula = diamonds$lprice ~ ., data = diamonds)
# 
# Residuals:
#   Min       1Q   Median       3Q      Max 
# -1.57080 -0.07681  0.00210  0.07690  2.19134 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)  -2.612e+00  7.099e-02  -36.79   <2e-16 ***
#   carat        -1.636e+00  7.766e-03 -210.67   <2e-16 ***
#   cutGood       4.948e-02  3.688e-03   13.41   <2e-16 ***
#   cutIdeal      1.070e-01  3.657e-03   29.25   <2e-16 ***
#   cutPremium    7.983e-02  3.495e-03   22.84   <2e-16 ***
#   cutVery Good  7.374e-02  3.571e-03   20.65   <2e-16 ***
#   colorE       -4.552e-02  1.929e-03  -23.59   <2e-16 ***
#   colorF       -8.169e-02  1.953e-03  -41.84   <2e-16 ***
#   colorG       -1.365e-01  1.921e-03  -71.08   <2e-16 ***
#   colorH       -2.047e-01  2.079e-03  -98.46   <2e-16 ***
#   colorI       -2.990e-01  2.381e-03 -125.59   <2e-16 ***
#   colorJ       -3.889e-01  3.025e-03 -128.54   <2e-16 ***
#   clarityIF     8.079e-01  6.042e-03  133.71   <2e-16 ***
#   claritySI1    3.877e-01  5.014e-03   77.34   <2e-16 ***
#   claritySI2    2.722e-01  4.897e-03   55.58   <2e-16 ***
#   clarityVS1    5.549e-01  5.262e-03  105.45   <2e-16 ***
#   clarityVS2    5.031e-01  5.136e-03   97.95   <2e-16 ***
#   clarityVVS1   7.337e-01  5.598e-03  131.08   <2e-16 ***
#   clarityVVS2   6.652e-01  5.460e-03  121.83   <2e-16 ***
#   depth         3.274e-02  1.035e-03   31.65   <2e-16 ***
#   table         1.121e-02  3.153e-04   35.53   <2e-16 ***
#   price         5.275e-05  4.664e-07  113.11   <2e-16 ***
#   x             8.375e-01  1.118e-02   74.90   <2e-16 ***
#   y             2.637e-01  1.144e-02   23.05   <2e-16 ***
#   z             5.903e-01  1.590e-02   37.12   <2e-16 ***
#   ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# Residual standard error: 0.1217 on 53891 degrees of freedom
# Multiple R-squared:  0.9856,	Adjusted R-squared:  0.9856 
# F-statistic: 1.539e+05 on 24 and 53891 DF,  p-value: < 2.2e-16


#Now that we are satisfying out requirement for Normally distributed data

plot(fit_log)

AIC(fit_log) 

#> AIC(fit_full)
#[1]  -74097.6


#Now we still hve some outliers lets look at them
# Lets see if our residuaks are distributed around zero
plot(residuals(fit_log),fit_log$fitted.values,main="Residuals vs Fitted Values")

identify(plot(residuals(fit_log),fit_log$fitted.values,main="Residuals vs Fitted Values"),plot=TRUE)

names(residuals(fit_log))

plot(residuals(fit_log)[residuals(fit_log)>2])

plot(residuals(fit_log)[residuals(fit_log)>-2])

identify(plot(residuals(fit_log)[residuals(fit_log)>2]))




diamonds[48411,]
diamonds[24068,]
diamonds[27416,]


#With outliers
#Residual standard error: 0.1251 on 53895 degrees of freedom
#Multiple R-squared:  0.9848,	Adjusted R-squared:  0.9848 
#F-statistic: 1.455e+05 on 24 and 53895 DF,  p-value: < 2.2e-16

#without outliers

#Residual standard error: 0.1217103 on 53892 degrees of freedom
#Multiple R-squared:  0.9856147,	Adjusted R-squared:  0.9856083 
#F-statistic: 153851.1 on 24 and 53892 DF,  p-value: < 0.00000000000000022204

#Based on the crrelation matrix can we do with table and depth and we defiitely don't want to use price to predict lprice

fit_unlimited <- lm(diamonds$lprice~ carat+cut+color+clarity+x+y+z+table+depth, data=diamonds)

summary(fit_unlimited)

plot(fit_unlimited);

# Before drop  outliers
#Residual standard error: 0.1354 on 53892 degrees of freedom
#Multiple R-squared:  0.9822,	Adjusted R-squared:  0.9822 
#F-statistic: 1.293e+05 on 23 and 53892 DF,  p-value: < 2.2e-16


# Try without table and depth - depth is a product of x,y,z
fit_limited<-lm(diamonds$lprice~ carat+cut+color+clarity+x+y+z ,data=diamonds)

summary(fit_limited)

plot(fit_limited)



########################################################## Stepwise Regression #####################################################



#Try AIC on the unlimited predictors for lprice
step(lm(diamonds$lprice~ carat+cut+color+clarity+x+y+z+table+depth, data=diamonds),direction="backward")


# Start:  AIC=-215623.8
# diamonds$lprice ~ carat + cut + color + clarity + x + y + z + 
#   table + depth
# 
# Df Sum of Sq     RSS     AIC
# <none>                  987.36 -215624
# - z        1     13.29 1000.65 -214905
# - y        1     14.20 1001.56 -214856
# - table    1     14.62 1001.98 -214833
# - depth    1     17.87 1005.24 -214659
# - cut      4     32.88 1020.25 -213865
# - x        1     68.27 1055.64 -212021
# - carat    1    498.73 1486.09 -193581
# - color    6    885.83 1873.19 -181110
# - clarity  7   1754.98 2742.34 -160561
# 
# Call:
#   lm(formula = diamonds$lprice ~ carat + cut + color + clarity + 
#        x + y + z + table + depth, data = diamonds)
# 
# Coefficients:
#   (Intercept)         carat       cutGood      cutIdeal    cutPremium  cutVery Good        colorE        colorF        colorG  
# -2.876373     -1.026948      0.074864      0.146546      0.117708      0.106317     -0.056546     -0.095829     -0.161662  
# colorH        colorI        colorJ     clarityIF    claritySI1    claritySI2    clarityVS1    clarityVS2   clarityVVS1  
# -0.256252     -0.376815     -0.514350      1.087041      0.579579      0.413659      0.794560      0.726564      0.995428  
# clarityVVS2             x             y             z         table         depth  
# 0.924049      0.757771      0.353506      0.475554      0.009903      0.035933 



step(lm(diamonds$lprice~ carat+cut+color+clarity+x+y+z+table+depth, data=diamonds), direction="forward")


# Start:  AIC=-215623.8
# diamonds$lprice ~ carat + cut + color + clarity + x + y + z + 
#   table + depth
# 
# 
# Call:
#   lm(formula = diamonds$lprice ~ carat + cut + color + clarity + 
#        x + y + z + table + depth, data = diamonds)
# 
# Coefficients:
#   (Intercept)         carat       cutGood      cutIdeal    cutPremium  cutVery Good        colorE        colorF        colorG  
# -2.876373     -1.026948      0.074864      0.146546      0.117708      0.106317     -0.056546     -0.095829     -0.161662  
# colorH        colorI        colorJ     clarityIF    claritySI1    claritySI2    clarityVS1    clarityVS2   clarityVVS1  
# -0.256252     -0.376815     -0.514350      1.087041      0.579579      0.413659      0.794560      0.726564      0.995428  
# clarityVVS2             x             y             z         table         depth  
# 0.924049      0.757771      0.353506      0.475554      0.009903      0.035933  



step(lm(diamonds$lprice~ carat+cut+color+clarity+x+y+z+table+depth, data=diamonds), direction="both")


# Start:  AIC=-215623.8
# diamonds$lprice ~ carat + cut + color + clarity + x + y + z + 
#   table + depth
# 
# Df Sum of Sq     RSS     AIC
# <none>                  987.36 -215624
# - z        1     13.29 1000.65 -214905
# - y        1     14.20 1001.56 -214856
# - table    1     14.62 1001.98 -214833
# - depth    1     17.87 1005.24 -214659
# - cut      4     32.88 1020.25 -213865
# - x        1     68.27 1055.64 -212021
# - carat    1    498.73 1486.09 -193581
# - color    6    885.83 1873.19 -181110
# - clarity  7   1754.98 2742.34 -160561
# 
# Call:
#   lm(formula = diamonds$lprice ~ carat + cut + color + clarity + 
#        x + y + z + table + depth, data = diamonds)
# 
# Coefficients:
#   (Intercept)         carat       cutGood      cutIdeal    cutPremium  cutVery Good        colorE        colorF        colorG  
# -2.876373     -1.026948      0.074864      0.146546      0.117708      0.106317     -0.056546     -0.095829     -0.161662  
# colorH        colorI        colorJ     clarityIF    claritySI1    claritySI2    clarityVS1    clarityVS2   clarityVVS1  
# -0.256252     -0.376815     -0.514350      1.087041      0.579579      0.413659      0.794560      0.726564      0.995428  
# clarityVVS2             x             y             z         table         depth  
# 0.924049      0.757771      0.353506      0.475554      0.009903      0.035933  


#Stepwise for fit_unliminted

null=lm(lprice~1, data=diamonds)
null

step(null, scope=list(lower=null, upper=fit_unlimited), direction="forward")


step(fit_unlimited, data=diamonds, direction="backward")

#stepwise regression both
step(null, scope = list(upper=fit_unlimited), data=diamonds, direction="both")



AIC(fit_unlimited)


#Stepwise for fit_liminted

null=lm(lprice~1, data=diamonds)
null


step(null, scope=list(lower=null, upper=fit_limited), direction="forward")


step(fit_limited, data=diamonds, direction="backward")

#stepwise regression both
step(null, scope = list(upper=fit_limited), data=diamonds, direction="both")



AIC(fit_limited)


#> AIC(fit_limited)
#[1]  -61123.75








# Residual standard error: 0.1377253 on 53896 degrees of freedom
# Multiple R-squared:  0.9815785,	Adjusted R-squared:  0.9815713 
# F-statistic: 136753.3 on 21 and 53896 DF,  p-value: < 0.00000000000000022204



#again after all outliers
#Residual standard error: 0.1353827 on 53893 degrees of freedom
#Multiple R-squared:  0.9822008,	Adjusted R-squared:  0.9821933 
#F-statistic: 129302.1 on 23 and 53893 DF,  p-value: < 0.00000000000000022204


#Residual standard error: 0.1372443 on 53895 degrees of freedom
#Multiple R-squared:  0.9817073,	Adjusted R-squared:  0.9817002 
#F-statistic: 137731.6 on 21 and 53895 DF,  p-value: < 0.00000000000000022204


# anova(fit_limited, fit_full)
# 
# > anova(fit_limited, fit_full)
# Analysis of Variance Table
# 
# Model 1: diamonds$lprice ~ carat + cut + color + clarity + x + y + z
# Model 2: diamonds$lprice ~ carat + cut + color + clarity + depth + table + 
#   price + x + y + z
# Res.Df        RSS Df Sum of Sq         F                 Pr(>F)    
# 1  53895 1015.16564                                                  
# 2  53892  798.32346  3 216.84217 4879.4167 < 0.000000000000000222 ***
#   ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# > 

fit_limited<-lm(diamonds$lprice~ carat+cut+color+clarity+x+y+z ,data=diamonds)

#Lets examine the plots for the Residual vs the predictor- how are they distributed

plot(fit_limited$residuals~diamonds$carat)
plot(fit_limited$residuals~diamonds$cut)
plot(fit_limited$residuals~diamonds$color)
plot(fit_limited$residuals~diamonds$clarity)
plot(fit_limited$residuals~diamonds$x)
plot(fit_limited$residuals~diamonds$y)
plot(fit_limited$residuals~diamonds$z)




#Residual vs fitted
plot(predict(fit_limited),residuals(fit_limited))
plot(predict(fit_limited),rstudent(fit_limited))
plot(hatvalues(fit_limited))
which.max(hatvalues(fit_limited))





fit <- aov(lprice ~ carat,  data=diamonds)
summary(fit)

fit <- aov(lprice ~ carat + cut,  data=diamonds)
summary(fit)
fit <- aov(lprice ~ carat + cut + color,  data=diamonds)
summary(fit)
fit <- aov(lprice ~ carat + cut + color + clarity,  data=diamonds)
summary(fit)
plot(fit) 


fit <- aov(lprice ~ x,  data=diamonds)
summary(fit)
fit <- aov(lprice ~ x + y ,  data=diamonds)
summary(fit)
fit <- aov(lprice ~ x + y+ z,  data=diamonds)
summary(fit)


qqplot(diamonds$carat,diamonds$lprice)
abline()
qqplot(diamonds$x,diamonds$lprice)

######Lets just see if we use only single predictor Carat
lm.fit=lm(lprice~ carat,data = diamonds)
summary(lm.fit)
confint(lm.fit)
predict(lm.fit,data.frame(carat=c(5,10,15)),interval="confidence")
predict(lm.fit,data.frame(carat=c(5,10,15)),interval="prediction")
plot(diamonds$carat,diamonds$lprice)
abline(lm.fit)

# Use Price
lm.fit=lm(price~ carat,data = diamonds)
summary(lm.fit)
confint(lm.fit)
predict(lm.fit,data.frame(carat=c(5,10,15)),interval="confidence")
predict(lm.fit,data.frame(carat=c(5,10,15)),interval="prediction")
plot(diamonds$carat,diamonds$price)
abline(lm.fit)

#So both still show price ceiling




########################################## Get a Handle on these outliers #########################################

if (!require("car")) install.packages("car")

library(car)

# (outliers_main_model <- influencePlot(fit))
# my_outlier<-c(outlierTest(fit))
# 
# #Sub-grade
# (outliers_main_model <- influencePlot(fit_a))
# my_outlier<-c(outlierTest(fit_a))
# 
# 
# my_outlier<-c(outliers_main_model)
# 
# (my_outlier)
# 
# 
# outliers_main_model
# 
# old_rownames<-row.names(x)
# 
# rownames(x) <- seq(length=nrow(x)) 
# 
# #Outlier identified from graph
# #Loan company employee Current Loan at 6% grade C c5 loan????
# x[65306,]

diamond_outliers<-influencePlot(fit_limited)

my_diamond_outliers<-c(diamond_outliers)

diamond_outliers

#So out data keep showing up these same outliers - about time we dealt with these.

# Reindex the dataframe
old_rownames<-row.names(diamonds)
rownames(diamonds) <- seq(length=nrow(diamonds)) 

#check value

diamonds[20686,]
diamonds[2274,]

qqplot


#diamonds[49173,]

diamonds[diamonds$cut=='Ideal',]

par(mfrow=c(1,1)) # two plots on one page

plot(as.numeric(row.names(diamonds)),diamonds$log_price)

summary(diamonds$price)
summary(diamonds$carat)
summary(diamonds$depth)
summary(diamonds$table)
summary(diamonds$x)
summary(diamonds$y)
summary(diamonds$z)

diamonds[(diamonds$carat==0.57) & (diamonds$cut=='Ideal') & (diamonds$color=='G'),]


############################################Create Trainng and Test Subsets 
n = nrow(diamonds)
trainIndex = sample(1:n, size = round(0.7*n), replace=FALSE)
diamonds_train = diamonds[trainIndex ,]
diamonds_test = diamonds[-trainIndex ,]


summary(diamonds_train)


summary(diamonds_test)

#### Drop price from dataset

diamonds_train$price = NULL

summary(diamonds_train)   


#diamonds_test$price = NULL

#summary(diamonds_test)   



####Apply Model from Training data to Test data
mod<-lm(lprice~carat+cut+color+clarity+x+y+z,data=diamonds_train)

output <- predict(mod, data.frame(diamonds_test))

output
output_price <-exp(output)
head(output_price)

summary(output_price)

plot(as.numeric(output_price),as.numeric(diamonds_test$price))


output_price_column=c(as.numeric(output_price))


                  
output_test_price_column =c(as.numeric(diamonds_test$price))


output_dataframe= data.frame(output_test_price_column,output_price_column)

cor(output_dataframe$output_test_price_column,output_dataframe$output_price_column)



summary(output_dataframe$output_test_price_column,output_dataframe$output_price_column)




head (output_dataframe)
summary(output_dataframe)

abline(as.numeric(output_price),as.numeric(diamonds_test$price))
str(output_price)
str(diamonds_test)
################################ Additional Models #######################


################################################## # Regression Tree Example to see if its any help #########################################
# Regression Tree Example
library(rpart)

# grow tree

fit<-rpart(diamonds$lprice~carat+cut+color+clarity+x+y+z+table+depth,method="anova",data=diamonds)


printcp(fit) # display the results
plotcp(fit) # visualize cross-validation results
summary(fit) # detailed summary of splits

# create additional plots
par(mfrow=c(1,2)) # two plots on one page
rsq.rpart(fit) # visualize cross-validation results  

# plot tree
plot(fit, uniform=TRUE,
     main="Regression Tree for LPrice ")
text(fit, use.n=TRUE, all=TRUE, cex=.8)

# create attractive postcript plot of tree
#post(fit, file = "c:/tree2.ps",
#     title = "Regression Tree for LPrice ")



#y is the key to branching

####################################Classification Models#######################################

summary(diamonds)

#Now lets try some Classification Models

#We can start with a simplee Example - divide the dataset into Ideal and non-Ideal cuts

#mins<- apply(data,2,min)
#maxs<- apply(data,2,max)

levels(diamonds$cut)

diamonds$ideal_flag <- ifelse(diamonds$cut =='Ideal', "True", "False")


tail(diamonds)

####################################### SVM #######################################



###### Create a Diamonds Matrix to expand Dummy variable for factors ####

names(diamonds)

carat + cut + color + clarity + depth + table+ x+y+z+lprice

#Create a Diamonds Matrix to expand Dummy variable for factors for SVM using cut

diamonds_matrix <- model.matrix( ~ cut
                                 + carat 
                                 + color 
                                 + clarity 
                                 + depth 
                                 + table
                                 + x
                                 +y
                                 +z
                                 +lprice  , data = diamonds)

diamonds_data=as.data.frame((diamonds_matrix))

str(diamonds_data)

#So lets take less than half the data say 10000 rows

library(dplyr)
diamond_sample <- sample_n(diamonds_data, 10000)

#put the 20000 sample back in diamonds data
diamonds_data=diamond_sample





library(e1071)

#diamonds_data<-subset(diamonds, select=-c(price,ideal_flag))

summary(diamonds_data)

svm_model<-svm(diamonds_data$ideal_flag~.,data=diamonds_data)

plot(svm_model,diamonds_data)



mins<- apply(diamonds_data,2,min)
maxs<- apply(diamonds_data,2,max)


diamonds_data.scaled <- as.data.frame(scale(diamonds_data, center = mins, scale = maxs - mins))
diamonds_data.index<-sample(1:nrow(diamonds_data),round(0.66*nrow(diamonds_data)))
diamonds_data.scaled$Type<-paste("T",diamonds_data$Type,sep="")

model <- svm(diamonds_data.scaled$Type~., type = "C-classification", data=diamonds_data.scaled[,-10] )

diamonds_data.svm.pred<-predict(model,diamonds_data.scaled[-diamonds_data.index,-10])

table(diamonds_data.svm.pred, diamonds_data[-diamonds_data.index,10])

quit("yes")






###### Create a Diamonds Matrix to expand Dummy variable for factors ####

names(diamonds)

carat + cut + color + clarity + depth + table+ x+y+z+lprice

#Create a Diamonds Matrix to expand Dummy variable for factors

diamonds_matrix <- model.matrix( ~ carat 
                                 + cut 
                                 + color 
                                 + clarity 
                                 + depth 
                                 + table
                                 + x
                                 +y
                                 +z
                                 +lprice , data = diamonds)

diamonds_data=as.data.frame((diamonds_matrix))

str(diamonds_data)

#So lets take less than half the data say 10000 rows

library(dplyr)
diamond_sample <- sample_n(diamonds_data, 10000)

#put the 20000 sample back in diamonds data
diamonds_data=diamond_sample





########################################################## Hierarchical Cluster Analysis #####################################################

help(memory.size)

memory.limit()

###Running into memory limit

#so lets subset the data and stratify and try this

names(diamonds_data)




#Hierarchical Cluster Analysis

#diamonds_data<-subset(diamonds, select=-c(price,ideal_flag))

#Using distance matrix cluster analysis for relationship discovery. Use hclust, and plot a dendrogram for attributes.
diamonds_dist <- dist(as.matrix(diamonds_data))   # find distance matrix 
diamonds_hc <- hclust(diamonds_dist)                # apply hirarchical clustering 
plot(diamonds_hc,cex=0.3,color="red")                       # plot the dendrogram 

########################################### Cluster ##############################


require(cluster)
library (cluster)

#diamonds_data<-subset(diamonds, select=-c(cut,price))
#diamonds_data<-subset(diamonds, select=-c(price,ideal_flag))

mins<- apply(diamonds_data,2,min)
maxs<- apply(diamonds_data,2,max)


diamonds_data.scaled <- as.data.frame(scale(diamonds_data, center = mins, scale = maxs - mins))


daisy(as.matrix(diamonds_data))


plot(daisy(as.matrix(diamonds_data)))     

plot(clusters.bclust(diam_dist))
     
library(bclust)

plot(bclust(diam_dist))
     
plot(daisy(diamonds_data))
     
     
     
########################################################## k-means Number of Clusters #####################################################
     
#    k-means
     
#diamonds_data<-subset(diamonds, select=-c(cut,price))

#diamonds_data<-subset(diamonds, select=-c(price,ideal_flag))
     

#diamonds_scaled <- scale(diamonds_data) # standardize variables 

diamonds_scaled <- (diamonds_data) # standardize variables 
     

is.na(diamonds_scaled)

diamonds_scaled <- scale(as.data.frame(diamonds_data)) # standardize variables 
     
summary(diamonds_scaled)
     
     
# Determine number of clusters
wss <- (nrow(diamonds_scaled)-1)*sum(apply(diamonds_scaled,2,var))

for (i in 2:15) wss[i] <- sum(kmeans(diamonds_scaled,
                                          centers=i)$withinss)

plot(1:15, wss, type="b", xlab="Number of Clusters",
  ylab="Within groups sum of squares") 
     

########################################################## K-Means Cluster Analysis #####################################################
     
     
# K-Means Cluster Analysis
fit <- kmeans(diamonds_scaled, 6) # 6 cluster solution
# get cluster means
aggregate(diamonds_scaled,by=list(fit$cluster),FUN=mean)
# append cluster assignment
diamonds_scaled <- data.frame(diamonds_scaled, fit$cluster) 

plot(diamonds_scaled)

     
########################################################## Ward Hierarchical Clustering #####################################################
     
# Ward Hierarchical Clustering
d_euclid_diam <- dist(diamonds_scaled, method = "euclidean") # distance matrix
fit_clust_diam <- hclust(d_euclid_diam, method="ward.D")
plot(fit_clust_diam,cex=0.1,main="Ward Hierarchical Clustering") # display dendogram


groups <- cutree(fit_clust_diam, k=5) # cut tree into 5 clusters
# draw dendogram with red borders around the 5 clusters
rect.hclust(fit_clust_diam, k=5, border="red") 
summary(fit_clust_diam)
     
########################################################## Model Based Clustering #####################################################
     
# Model Based Clustering
library(mclust)
fit_mod_clust <- Mclust(diamonds_scaled)
plot(fit_mod_clust ,main="Model Based Clustering") # plot results
summary(fit_mod_clust ) # display the best model 



######################################## GLM #####################################

data<-diamonds

index <- sample(1:nrow(data),round(0.75*nrow(data)))
train_diamond <- data[index,]
test_diamond <- data[-index,]
lm.fit <- glm(lprice~., data=train_diamond)
summary(lm.fit)
pr.lm <- predict(lm.fit,test_diamond)

summary(lm.fit)
summary(pr.lm)

MSE.lm <- sum((pr.lm - test$lprice)^2)/nrow(test)

summary(MSE.lm)

#create a model matrix for the glm model
mm<-model.matrix(glm(lprice~., data=train_diamond))

mm$lprice<-diamonds$lprice
#Remove Space in Model Matrix d0esn't work so need manual solution
#mm <- make.names(mm, unique=TRUE)

head(mm)

(mm[,1])

colnames(mm)

tail(mm[,c(0,1,2,3,4,5,6,7,8,9,10,11)],n=5)

colnames(mm)[6]

#A bit of renaming to help with out formula creation
colnames(mm)[6]<-"cutVeryGood"
colnames(mm)[1]<-"Intercept"
colnames(mm)

mm[,6]
#maxs <- apply(data, 2, max) 
#mins <- apply(data, 2, min)

#scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))

#Need to add in the lprice data

mm<-cbind(mm, data.matrix(train_diamond$lprice))

#rename the log_price column

#Lets figure uot which it is
head(mm[],n=5)
mm[1, 27]

colnames(mm)[27]<-"lprice"





#check data is correct 
head(train_diamond$lprice)
#looks good so do a subtratcion and see if we get zero
tail(mm)
tail(train_diamond$lprice)

summary(mm["lprice"])

#Check for NA
unique(is.na(train_diamond))

summary(train_diamond)

maxs <- apply(mm, 2, max) 
mins <- apply(mm, 2, min)

scaled <- as.data.frame(scale(mm, center = mins, scale = maxs - mins))

summary(scaled)

train_ <- scaled[index,]
test_ <- scaled[-index,]

head(train_)

is.na(train_)
unique(is.na(train_))
################################ Neural network ##########################
library(neuralnet)
n <- names(train_)
f <- as.formula(paste("lprice ~", paste(n[!n %in% "lprice"], collapse = " + ")))
nn <- neuralnet(f,data=train_,hidden=c(5,3),linear.output=T)

#error 
summary(train_)

#Need to look again at the field in use price is in there- that can go -- ideal_flag can go

nn<-NULL

#Drop the intecept and price
nn <- neuralnet(lprice ~ carat + cutGood + cutIdeal + cutPremium + 
                  cutVeryGood + colorE + colorF + colorG + colorH + colorI + 
                  colorJ + clarityIF + claritySI1 + claritySI2 + clarityVS1 + 
                  clarityVS2 + clarityVVS1 + clarityVVS2 + depth + table + 
                  x + y + z + ideal_flagTrue,data=train_,hidden=c(5,3),linear.output=T)



plot(nn)

################################################## Write data after munging ##################

write.csv(diamonds, file = "diamonds_munged.csv")
write.csv(train_diamond, file = "diamonds_train.csv")
write.csv(test_diamond, file = "diamonds_test.csv")


################################################## Redundant Code ##################


plot(diamonds$price,diamonds$carat)
plot(diamonds$carat,diamonds$price)
plot(diamonds$x,diamonds$price)
plot(diamonds$y,diamonds$price)
plot(diamonds$z,diamonds$price)

diamonds[diamonds$x<1,]
diamonds[diamonds$y<1,]
diamonds[diamonds$z<1,]


#Check the y and z value for outliers

summary(diamonds$y) # max = 58.9

summary(diamonds$z) #max 31.8

#> summary(diamonds$y)
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#3.680   4.720   5.710   5.735   6.540  58.900 

#> summary(diamonds$z)
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#1.07    2.91    3.53    3.54    4.04   31.80 

#Examine these outlier values

diamonds[diamonds$y >55,]

#diamonds[diamonds$y >55,]
#carat     cut color clarity depth table price    x    y    z log_price
#24068     2 Premium     H     SI2  58.9    57 12210 8.09 58.9 8.06  9.410011


diamonds[diamonds$z >30,]
#carat       cut color clarity depth table price    x    y    z log_price
#48411  0.51 Very Good     E     VS1  61.8  54.7  1970 5.12 5.15 31.8  7.585789






