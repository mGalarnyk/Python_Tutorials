import re
str='Nagendra rao: 9706.612234'
res=re.search(r'\d+',str)
print(res.group())