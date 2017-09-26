<h1 align="center"> Scrapy for Data Analysis Talk</h1>

The first part of this tutorial is highly similar to the [official scrapy documentation](https://doc.scrapy.org/en/latest/intro/tutorial.html) has been tested in Python 2 and 3 (work in both).

You can see this code in action by clicking on the following link: [youtube video](https://youtu.be/O_j3OTXw2_E). 

## Getting Started (Prerequisites)

### If you already have anaconda and google chrome (or Firefox), skip to step 4.

1. Install Anaconda (Python) on your operating system. You can either download anaconda from the official site and install on your own or you can follow these anaconda installation tutorials below.

Operating System | Blog Post | Youtube Video
--- | --- | ---
Mac | [Install Anaconda on Mac](https://medium.com/@GalarnykMichael/install-python-on-mac-anaconda-ccd9f2014072#.lvhw2gt3k "Install Anaconda on Mac") | [Youtube Video](https://www.youtube.com/watch?v=B6d5LrA8bNE "Youtube Video")
Windows | [Install Anaconda on Windows](https://medium.com/@GalarnykMichael/install-python-on-windows-anaconda-c63c7c3d1444#.66f7y3whf) | [Youtube Video](https://www.youtube.com/watch?v=dgjEUcccRwM)
Ubuntu | [Install Anaconda on Ubuntu](https://medium.com/@GalarnykMichael/install-python-on-ubuntu-anaconda-65623042cb5a#.4kwsp0wjl) | [Youtube Video](https://www.youtube.com/watch?v=jo4RMiM-ihs)
All | [Environment Management with Conda (Python 2 + 3, Configuring Jupyter Notebooks)](https://medium.com/towards-data-science/environment-management-with-conda-python-2-3-b9961a8a5097) | [Youtube Video](https://www.youtube.com/watch?v=rFCBiP9Gkoo)

2. Install Scrapy (anaconda comes with it, but just in case). You can also install on your terminal (mac/linux) or command line (windows). 

```
conda install scrapy
```

3. Make sure you have Google chrome or Firefox. In this tutorial I am using Google Chrome. If you dont have google chrome and want to install it, you can either google it or install it here using this [link](https://support.google.com/chrome/answer/95346?co=GENIE.Platform%3DDesktop&hl=en).

## Creating a new Scrapy project

1. Open a terminal (mac/linux) or command line (windows).  Navigate to a desired folder (see the image below if you need help) and type 

```
scrapy startproject fundrazr
```

![](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Scrapy/Tutorial_Images/startProject.png)

<br>

This makes a fundrazr directory with the following contents:

![](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Scrapy/Tutorial_Images/fundrazrProjectDirectory.png)

## Finding good start URLs using Inspect on Google Chrome (or Firefox)
In the spider framework, <b>start_urls</b> is a list of URLs where the spider will begin to crawl from, when no particular URLs are specified. We will use each element in the <b>start_urls</b> list as a means to get individual campaign links. 

1. The image below shows that based on the category you choose, you get a different start url. The highlighted part in black are the possible categories of fundrazrs to scrape.    

![](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Scrapy/Tutorial_Images/StartUrlsFundrazr.png)
<br>

For this tutorial, the first in the list <b>start_urls</b> is: https://fundrazr.com/find?category=Health

2. This part is about getting additional elements to put in the <b>start_urls</b> list. We are finding out how to go to the next page so we can get additional urls to put in <b>start_urls</b>. 

![](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Scrapy/Tutorial_Images/inspectNextFigure.png)
<br>

The second start url is: https://fundrazr.com/find?category=Health&page=2

The code below will be used in the spider later. All it does is make a list of start_urls. The variable npages is just how many additional pages (after the first page) we want to get campaign links from. 

```
# First Start Url
start_urls = ["https://fundrazr.com/find?category=Health"]

npages = 2

# This mimics getting the pages using the next button. 
for i in range(2, npages + 2 ):
	start_urls.append("https://fundrazr.com/find?category=Health&page="+str(i)+"")
```

## Scrapy Shell for finding Individual Campaign Links
The best way to learn how to extract data with Scrapy is using the Scrapy shell. We will use XPaths which can be used to select elements from HTML documents. 

The first thing we will try and get the xpaths for are the individual campaign links. First we do inspect to see roughly where the campaigns are in the HTML. 

![](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Scrapy/Tutorial_Images/inspectCampaigns.png)
<br>

We will use xpaths to extract the part enclosed in the red rectangle below.  

![](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Scrapy/Tutorial_Images/campaignLink.png)
<br>

The best way to to make xpaths and to check if they work is to test it inside scrapy shell. 

In terminal type (mac/linux): 

```
scrapy shell 'https://fundrazr.com/find?category=Health'
```

In command line type (windows): 

```
scrapy shell "https://fundrazr.com/find?category=Health"
```

Type the following into scrapy shell (to help understand the code, please see the video):

```
response.xpath("//h2[contains(@class, 'title headline-font')]/a[contains(@class, 'campaign-link')]//@href").extract()
```

![](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Scrapy/Tutorial_Images/individualCampaignLinks.png)
<br>


The code below is for getting all the campaign links for a given start url (more on this later in the First Spider section) 

```
for href in response.xpath("//h2[contains(@class, 'title headline-font')]/a[contains(@class, 'campaign-link')]//@href"):
	# add the scheme, eg http://
	url  = "https:" + href.extract() 
```

2. Exit Scrapy Shell by typing <b>exit()</b>. We do this while we should now understand the structure of where individual campaigns links are, we havent looked at where things are on individual campaigns.

![](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Scrapy/Tutorial_Images/exitScrapyShell.png)
<br>

## Inspecting Individual Campaigns

1. Next we go to an individual campaign page (see link below) to scrape (I should note that some of these campaigns are difficult to view)

https://fundrazr.com/savemyarm

2. On the page using the same inspect process as before we inspect the title on the page

![](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Scrapy/Tutorial_Images/InspectCampaignTitle.png)
<br>

3. Now we are going to use scrapy shell again, but this time with an individual campaign. We do this because we want to find out how individual campaigns are formatted. 

In terminal type (mac/linux): 

```
scrapy shell 'https://fundrazr.com/savemyarm'
```

In command line type (windows): 

```
scrapy shell "https://fundrazr.com/savemyarm"
```

The code to get the campaign title is

```
response.xpath("//div[contains(@id, 'campaign-title')]/descendant::text()").extract()[0]
```

![](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Scrapy/Tutorial_Images/GettingTitleIndividualCampaignShell.png)
<br>

4. We can do the same for the other parts of the page.

amount Raised: 

```
response.xpath("//span[contains(@class, 'stat')]/span[contains(@class, 'amount-raised')]/descendant::text()").extract()
```

goal: 
```
response.xpath("//div[contains(@class, 'stats-primary with-goal')]//span[contains(@class, 'stats-label hidden-phone')]/text()").extract()
```

currency type: 
```
response.xpath("//div[contains(@class, 'stats-primary with-goal')]/@title").extract()
```

campaign end date:
```
response.xpath("//div[contains(@id, 'campaign-stats')]//span[contains(@class,'stats-label hidden-phone')]/span[@class='nowrap']/text()").extract()
```

number of contributors: 
```
response.xpath("//div[contains(@class, 'stats-secondary with-goal')]//span[contains(@class, 'donation-count stat')]/text()").extract()
```

story: 
```
response.xpath("//div[contains(@id, 'full-story')]/descendant::text()").extract()
```

url: 
```
response.xpath("//meta[@property='og:url']/@content").extract()
```

5. Exit scrapy shell by typing: 

```
exit()
```

## Items
The main goal in scraping is to extract structured data from unstructured sources, typically, web pages. Scrapy spiders can return the extracted data as Python dicts. While convenient and familiar, Python dicts lack structure: it is easy to make a typo in a field name or return inconsistent data, especially in a larger project with many spiders.

![](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Scrapy/Tutorial_Images/items.png) 
<br>

The code for items.py is [here](https://github.com/mGalarnyk/Python_Tutorials/raw/master/Scrapy/fundrazr/fundrazr/items.py).

Save it under the fundrazr/fundrazr directory (overwrite the original iems.py file). 

The item class (basically how we store our data before outputting it) used in this tutorial looks like this. 

![](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Scrapy/Tutorial_Images/itemsFundrazr.png)
<br>

## The Spider 
Spiders are classes that you define and that Scrapy uses to scrape information from a website (or a group of websites). The code for our spider is below.

![](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Scrapy/Tutorial_Images/fundrazrScrapyCode.png)
<br>

Download the code [here](https://raw.githubusercontent.com/mGalarnyk/Python_Tutorials/master/Scrapy/fundrazr/fundrazr/spiders/fundrazr_scrape.py).

Save it in a file named <b>fundrazr_scrape.py</b> under the fundrazr/spiders directory.

The current project should now have the following contents:
![](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Scrapy/Tutorial_Images/DirectoryafterMakingFile.png)

## Running the Spider

1. go to the fundrazr/fundrazr directory and type: 

```
scrapy crawl my_scraper -o MonthDay_Year.csv
```

![](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Scrapy/Tutorial_Images/ScraperRunning.png)
<br>

2. The data should be outputted in the fundrazr/fundrazr directory. 

![](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Scrapy/Tutorial_Images/DataOutputtedLocation.png)
<br>

## Our data

1. The data outputted in this tutorial should look roughly like the image below. The individual campaigns scraped will vary as the website is constantly updated. Also it is possible there will be spaces between each individual campaign as excel is interpreting the csv file. 

![](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Scrapy/Tutorial_Images/WebScrapingData1st.png)
<br>

2. If you want to download a larger file (it was made by changing npages = 2 to npages = 450), you can download a bigger file with roughly 6000 campaigns scraped by clicking on this [link](https://github.com/mGalarnyk/Python_Tutorials/raw/master/Scrapy/fundrazr/fundrazr/MiniMorningScrape.csv)

![](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Scrapy/Tutorial_Images/dataset.png)
