# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy

class FundrazrItem(scrapy.Item):
	campaignTitle = scrapy.Field()
	amountRaised = scrapy.Field()
	goal = scrapy.Field()
	currencyType = scrapy.Field()
	endDate = scrapy.Field()
	numberContributors = scrapy.Field()
	story = scrapy.Field()
	url = scrapy.Field()
