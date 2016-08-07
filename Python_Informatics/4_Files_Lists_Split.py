# -*- coding: utf-8 -*-
"""
Michael Galarnyk
Assignment: Files, Lists, and Split

1. Write a program that prompts the user for a filename. Open the file, and read it one line at a time. For each line split the line into a list of words called line_list. For each word in the current line_list, look to see if it is in a list called script_list. If the word is not in script_list, add it to the script_list. Sort the script_list alphabetically.
2. Within the same program define a function called freq_count(). This function accepts a str and a list of words as arguments. It traverses the list of words and searches each word and counts the occurrences of the substring str within each word. Print each word along with the number of substring occurrences found with the associated word.
3. Test your program with the romeo.txt file that comes as a text file resource with our textbook. Your program should accept the filename and the substring str as input from the user. After reading the file to build and sort script_list, pass script_list into the freq_count() function.
"""

def freq_count(user_string, unique_words):
    for word in unique_words:
        print word, ":", user_string.count(word);
        
text_file_name = raw_input('Enter a text file please \n');
user_string = raw_input('What substring do you want ? \n')
with open('romeo.txt', 'r') as f:
    script_list = [];
    line_list = [];
    for line in f:
        line_list.extend(line.split());

for word in line_list:
    if word not in script_list:
        script_list.append(word);
         
script_list.sort();

freq_count(user_string, script_list);
