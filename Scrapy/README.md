<h1 align="center"> Scrapy for Data Analysis Talk</h1>

The first part of this tutorial is highly similar to the [official scrapy documentation](https://doc.scrapy.org/en/latest/intro/tutorial.html)

## Getting Started

### Prerequisites: Anaconda. If you already have anaconda and google chrome (or Firefox), skip to step 4.

1. Install Anaconda (Python) on your operating system. You can either download anaconda from the official site and install on your own or you can follow these anaconda installation tutorials below.

Windows: if you dont have 

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


