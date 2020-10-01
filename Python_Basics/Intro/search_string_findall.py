import re
str='man sun mop run'
result=re.findall(r'm\w\w',str)
print(result)
