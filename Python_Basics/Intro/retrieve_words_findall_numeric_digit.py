import re
str='the meeting will be conducted on 1st and 21st of every month'
res=re.findall(r'\d[\w]*',str)
print(res)