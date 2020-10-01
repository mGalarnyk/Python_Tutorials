import re
str='Rahul got 75 marks,Vijay got 55 marks,whereas Subbu got 98 marks.'

marks=re.findall(r'\d{2}',str)
print(marks)

names=re.findall(r'[A-Z][a-z]*',str)
print(names)