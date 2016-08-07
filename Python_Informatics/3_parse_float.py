# -*- coding: utf-8 -*-
"""
Michael Galarnyk

Given the following python statement…
avg_str = 'Average value read: 0.72903'
Use the find() method and string slicing
to extract the potion of the string after
the colon character and then use the float()
function to convert the extracted string 
into a floating point value. Save your code 
in a file named “parse_float.py”.
"""

avg_str = 'Average value read: 0.72903'
start_pos = avg_str.find(':');

# 1 more since start_po is after :
start_pos = start_pos + 1;

# taking the number portion of the string
number_portion = avg_str[start_pos:];

# typecasting number_portion to float
number_portion = float(number_portion)
