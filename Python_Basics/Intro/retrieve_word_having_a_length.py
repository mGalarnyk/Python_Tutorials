import re
str='one two three four five six seven 8 9 10'
#result=re.findall(r'\b\w{4,}\b',str)
#result=re.findall(r'\b\w{5}\b',str)
#result=re.findall(r'\b\w{3,5}\b',str)
result=re.findall(r'\b\d\b',str)
print(result)
#res=re.search(r'\b\w{5}\b',str)
#print(res.group())