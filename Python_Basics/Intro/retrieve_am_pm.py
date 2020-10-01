import re
str='the meeting may be at 8pm or 9am or 4pm or 5pm.'
res=re.findall(r'\dam|\dpm',str)
print(res)