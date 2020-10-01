import re
str='anil akhil anant arun arati arundhati abhijit ankur'
res=re.findall(r'a[nk][\w]*',str)
print(res)