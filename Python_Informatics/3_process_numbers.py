# -*- coding: utf-8 -*-
"""
Michael Galarnyk

Write a program called “process_numbers.py”
that repeatedly reads numbers input by 
the user until the user types “done”. 
After the user has entered “done”, 
print out the total, count, maximum,
 minimum, and average of the entered numbers.
"""
user_input = ''
total = 0;
count = 0;
maximum = None;
minimum = None; 
while (user_input != 'done'):
    user_input = raw_input('Enter a number or enter done\n');
    
    if user_input != 'done':
        total += float(user_input);
    if (user_input == 'done'):
        average = total / float(count);
        break; 
    count += 1
    if maximum is None or user_input > maximum:
        maximum = user_input
    if minimum is None or user_input < minimum:
        minimum = user_input
    

print 'count: %d \n' %count
print 'average%g \n' %average 
print 'maximum:%s \n' %maximum
print 'minimum:%s \n' %minimum 
