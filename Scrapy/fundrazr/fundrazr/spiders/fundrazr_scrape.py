import scrapy
from fundrazr.items import FundrazrItem
from datetime import datetime
import re


class Fundrazr(scrapy.Spider):
	name = "my_scraper"

	allowed_domains = ["https://fundrazr.com/"]

	# Potential Start Urls
	start_urls = ["https://fundrazr.com/find?category=Health"]

	# response.xpath('//html/body').css('.tile-img-contain').xpath('./a[1]/@href')

	npages = 50
	for i in range(2, npages +1 ):
		start_urls.append("https://fundrazr.com/find?category=Health&page="+str(i)+"")

	# getting individual campaigns
	def parse(self, response):
		for href in response.xpath("//h2[contains(@class, 'title headline-font')]/a[contains(@class, 'campaign-link')]//@href").extract():
			url = response.urljoin(href[2:].extract())
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

		# Stat for how long left (but there is no Label) need to work on
		# response.xpath("//div[contains(@class, 'stats-secondary with-goal')]//span[contains(@class,'stats-label visible-phone')]/span[@class='stats-label']/text()").extract()


		item = FundrazrItem()		
		yield item