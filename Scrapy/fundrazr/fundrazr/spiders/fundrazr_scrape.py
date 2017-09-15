import scrapy
from fundrazr.items import FundrazrItem
from datetime import datetime
import re


class Fundrazr(scrapy.Spider):
	name = "my_scraper"

	allowed_domains = ["https://fundrazr.com/"]

	# Potential Start Urls
	start_urls = ["https://fundrazr.com/find?category=Health"]

	npages = 50
	for i in range(2, npages +1 ):
		start_urls.append("https://fundrazr.com/find?category=Health&page="+str(i)+"")
	
	def parse(self, response):

	# getting individual campaigns
		for href in response.xpath("//h2[contains(@class, 'title headline-font')]/a[contains(@class, 'campaign-link')]//@href").extract():
			# add the scheme, eg http://
			url  = "https:" + href 

			yield scrapy.Request(url, callback=self.parse_dir_contents)	
					
	def parse_dir_contents(self, response):

		# Getting Campaign Title
		# = response.xpath("//div[contains(@id, 'campaign-title')]/descendant::text()").extract()[0]

		# Getting Amount Raised
		# response.xpath("//span[contains(@class, 'stat')]/span[contains(@class, 'amount-raised')]/descendant::text()").extract()

		# Currency Symbol
		# response.xpath("//span[contains(@class, 'stat')]/span[contains(@class, 'currency-symbol')]/descendant::text()").extract()

		# Raised Progress
		# response.xpath("//span[contains(@class, 'stats-label hidden-phone')]/span[contains(@class, 'raised-progress')]/descendant::text()").extract()[0]

		# Goal (will need to filter this)
		# response.xpath("//div[contains(@class, 'stats-primary with-goal')]//span[contains(@class, 'stats-label hidden-phone')]/text()").extract()

		# Number of contributors
		# response.xpath("//div[contains(@class, 'stats-secondary with-goal')]//span[contains(@class, 'donation-count stat')]/text()").extract()

		# Stat for how long left (but there is no Label) 
		# response.xpath("//div[contains(@class, 'stats-secondary with-goal')]//span[contains(@class,'stats-label visible-phone')]/span[@class='stat']/text()").extract()[0]

		# How long left unit like days months years etc
		# "".join(response.xpath("//div[contains(@class, 'stats-secondary with-goal')]//span[@class='stats-label visible-phone']/span[@class='stats-label']/text()").extract())

		# Non Mobile how long left as in exact date
		# "".join(response.xpath("//div[contains(@id, 'campaign-stats')]//span[contains(@class,'stats-label hidden-phone')]/span[@class='nowrap']/text()").extract())

		# Getting Story
		story_list = response.xpath("//div[contains(@id, 'full-story')]/descendant::text()").extract()
		#remove empty paragraph in story_obj
		story_list = [x.strip() for x in story_obj if len(x.strip()) > 0]

		story  = " ".join(story_list)
		print(story)


		item = FundrazrItem()		
		yield item