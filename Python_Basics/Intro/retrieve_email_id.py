import re
f=open('mails.txt','r')
str=f.read()
res=re.findall(r'\S+@\S+',str)
print(res)
f.close()