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
conda install -c conda-forge scrapy
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

## Choosing a Start Url 
<h1 align="center"> Scrapy Shell</h1>

1. Possible start url: https://fundrazr.com/find?category=Health

![](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Scrapy/Tutorial_Images/StartUrlsFundrazr.png)

2. Inspect using google chrome and such. 

## First Spider

5. (placeholder, will change when make code) This is the code for our first Spider. Save it in a file named quotes_spider.py under the fundrazr/spiders directory in your project:

![](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Scrapy/Tutorial_Images/1stSpiderPlaceholder.png)

6. Navigate to cd Code/Python/GoFundMe

7. Running the Spider (this needs to be updated fixed when make code)

```
scrapy crawl spidername -o MonthDay_Year.csv
```

<h1 align="center"> Things to add </h1>
name, allowed_domains, start_urls

