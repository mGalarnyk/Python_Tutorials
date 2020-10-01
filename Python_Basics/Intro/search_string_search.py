import re
str='man sun mop run'
#result=re.search(r'm\w\w',str)
result=re.findall(r'm\w\w',str)
print(result)
#print(result.group())#for search method