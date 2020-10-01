import re
str='this; is the: "core" python\'s book'
result=re.split(r'\W+',str)
print(result)