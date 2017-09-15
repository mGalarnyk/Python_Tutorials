<h1 align="center"> Scrapy for Data Analysis Talk</h1>

The first part of this tutorial is highly similar to the [official scrapy documentation](https://doc.scrapy.org/en/latest/intro/tutorial.html)

## Getting Started

### Prerequisites: Anaconda. If you already have anaconda and google chrome (or Firefox), skip to step 4.

1. Install Anaconda (Python) on your operating system. You can either download anaconda from the official site and install on your own or you can follow these anaconda installation tutorials below.

Operating System | Blog Post | Youtube Video
--- | --- | ---
Mac | [Install Anaconda on Mac](https://medium.com/@GalarnykMichael/install-python-on-mac-anaconda-ccd9f2014072#.lvhw2gt3k "Install Anaconda on Mac") | [Youtube Video](https://www.youtube.com/watch?v=B6d5LrA8bNE "Youtube Video")
Windows | [Install Anaconda on Windows](https://medium.com/@GalarnykMichael/install-python-on-windows-anaconda-c63c7c3d1444#.66f7y3whf) | [Youtube Video](https://www.youtube.com/watch?v=dgjEUcccRwM)
Ubuntu | [Install Anaconda on Ubuntu](https://medium.com/@GalarnykMichael/install-python-on-ubuntu-anaconda-65623042cb5a#.4kwsp0wjl) | [Youtube Video](https://www.youtube.com/watch?v=jo4RMiM-ihs)
All | [Environment Management with Conda (Python 2 + 3, Configuring Jupyter Notebooks)](https://medium.com/towards-data-science/environment-management-with-conda-python-2-3-b9961a8a5097) | [Youtube Video](https://www.youtube.com/watch?v=rFCBiP9Gkoo)

2. Install Scrapy (anaconda comes with it, but just in case). You can also install on your terminal (mac/linux) or command line (windows)
```
conda install scrapy
```

3. Make sure you have Google chrome or Firefox. In this tutorial I am using Google Chrome. If you dont have google chrome want to install it, you can install it here using this [link](https://support.google.com/chrome/answer/95346?co=GENIE.Platform%3DDesktop&hl=en).

4. Open a terminal (mac/linux) or command line (windows).  Navigate to a desired folder (see the image below if you need help) and type 

```
scrapy startproject fundrazr
```

![](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Scrapy/Tutorial_Images/startProject.png)

<br>

This Makes a fundrazr directory with the following contents:

![](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Scrapy/Tutorial_Images/fundrazrProjectDirectory.png)

## Finding good start URLs: Understanding Website Structure by Inspecting using Google Chrome 
The purpose of this is really just to find something to scrape. 

1. Possible start url: https://fundrazr.com/find?category=Health

![](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Scrapy/Tutorial_Images/StartUrlsFundrazr.png)
<br>
The highlighted part in black are the possible categories of fundrazrs to scrape. 

2. Finding out how to go to the next page. 

![](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Scrapy/Tutorial_Images/inspectNextFigure.png)
<br>
3. Finding out where the links to individual campaigns are on a page 

![](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Scrapy/Tutorial_Images/inspectCampaigns.png)

## Scrapy Shell for finding start URLs on Page
The best way to learn how to extract data with Scrapy is trying selectors using the shell Scrapy shell. 

In terminal type (mac/linux): 

```
scrapy shell 'https://fundrazr.com/find?category=Health'
```

In command line type (windows): 

```
scrapy shell "https://fundrazr.com/find?category=Health"
```

1. Our first learning is extracting links to individual campaigns. 

![](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Scrapy/Tutorial_Images/campaignLink.png)
<br>

The image below is inside scrapy shell

![](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Scrapy/Tutorial_Images/individualCampaignLinks.png)
<br>

This is the code that reflects getting all the campaign links (more on this later in the First Spider section) 
(need to do add in [2:] for all links since we start with // instead of instead of a normal url)

![](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Scrapy/Tutorial_Images/codeToGetCampaignLinks.png)
<br>

2. Exiting Scrapy Shell using <b>exit()</b>. We do this because now we want to understand the structure of an average campaign page. 

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

The code to get the title is

```
response.xpath("//div[contains(@id, 'campaign-title')]/descendant::text()").extract()[0]
```

![](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Scrapy/Tutorial_Images/GettingTitleIndividualCampaignShell.png)
<br>

4. We can do the same for the other parts of the page. <b>Need to put in exceptions after  </b> finding out rest of page

## Items
The main goal in scraping is to extract structured data from unstructured sources, typically, web pages. Scrapy spiders can return the extracted data as Python dicts. While convenient and familiar, Python dicts lack structure: it is easy to make a typo in a field name or return inconsistent data, especially in a larger project with many spiders.

![](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Scrapy/Tutorial_Images/items.png) 
<br>

1. To be added

## First Spider

![](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Scrapy/Tutorial_Images/1stSpiderPlaceholder.png)

1. (placeholder, will change when make code) This is the code for our first Spider. Save it in a file named fundrazr_scrape.py under the fundrazr/spiders directory in your project:

The current fundrazr directory should now have the following contents:
![](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Scrapy/Tutorial_Images/DirectoryafterMakingFile.png)

2. Navigate to cd Code/Python/GoFundMe

## Running the Spider (this needs to be updated fixed when make code)

1. go to the cd cd fundrazr/fundrazr
scrapy crawl my_scraper -o MonthDay_Year.csv

```
scrapy crawl my_scraper -o MonthDay_Year.csv
```

<h1 align="center"> Things to add </h1>
name, allowed_domains, start_urls

 pip install urllib2
