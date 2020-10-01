import re
str='kumbhmela will be conducted at Ahmedabad in India'
res=re.sub(r'Ahmedabad','Allahabad',str)
print(res)