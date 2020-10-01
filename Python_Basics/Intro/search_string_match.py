import re
str='man sun mop run'
result=re.match(r'm\w\w',str)
print(result.group())