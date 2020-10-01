import re
str='two three one two three'
res=re.findall(r't\w*\Z',str)
print(res)
