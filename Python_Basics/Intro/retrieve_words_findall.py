import re
str='an apple a day keeps the doctor away'
res=re.findall(r'\ba[\w]*\b',str)
print(res)