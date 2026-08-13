twitter-diversity
=================

Data files for the paper "Forecasting the pulse: How deviations from regular patterns in online data can identify offline phenomena"

There are two comma-separated files:

- tag-volume.csv contains 1000 data series consisting of the tweet volume of the 1000 most popular hashtags
- total-volume.csv contains a single data series of the total volume of tweets per hour

Both files include a date field that follows the ISO standard, ie: YYYYMMDDHH (year, month, day, hour).

These files can be easily imported into a statistics package such as R:

	tags <- read.csv("tag-volume.csv", header=TRUE)
	tags$date <- as.POSIXct(strptime(tags$date, format='%Y%m%d%H'))
	
Questions and suggestions can be submitted via pull request or directed at the authors:  

pascal.juergens@googlemail.com and  
andreas.jungherr@googlemail.com