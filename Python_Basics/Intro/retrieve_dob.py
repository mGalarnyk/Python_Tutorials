import re
#str='vijay 20 192.162.10.12 ,rohit 21 22-10-1990'
str='''we are connecting to server having IP 192.168.10.12 for handson
my email id is john@wipro.com use this email for official purpose.
CIS triaing going on well'''
res=re.search(r'\b\d{3}.\d{3}.\d{2}.\d{2}\b',str)
print(res.group())