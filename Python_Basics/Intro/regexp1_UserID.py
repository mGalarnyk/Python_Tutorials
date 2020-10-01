import re
f=open('reg1_UserID.txt','r')
str=f.read()
res=re.search(r'\b\d{3}.\d{3}.\d{2}.\d{2}\b',str)
print(res.group())
f.close()