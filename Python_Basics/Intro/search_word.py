import re
str='hello world'
res=re.search(r"world$",str,re.IGNORECASE)
if res:
    print("string ends with 'world'")
else:
    print('string does not ends with "world"')