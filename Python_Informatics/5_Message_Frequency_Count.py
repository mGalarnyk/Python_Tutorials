# -*- coding: utf-8 -*-
"""
Michael Galarnyk
Assignment: Message Frequency Count

1. Write a program that reads through the mail box data and when you find a line that starts with “From”, extract the address information from the line. Count the number of messages from each person by using a dictionary. Note that you might need to look at more than “From” because of duplicate instances of the address (hint: “From “ vs. “From:”).
2. After all of the data has been read, print the person with the highest number of messages. To do this, create a list of tuples (count, email) from the dictionary, sort the list in reverse order and print out the person who has the highest number of messages.
"""

d = {};

with open('Data_Files/mbox.txt', 'r') as f:

    for line in f:
        line = line.translate(None, '!\'#$%*:+')  # removing punctuation
        line = line.lower().split()
        try:
            if line[0] == 'from':
                d[line[1]] = d.get(line[1],0) + 1
        except:
            pass

message_freq = [];

for key, val in d.items():
    message_freq.append((val,key));

message_freq.sort(reverse=True)

print 'The person with the highest number of messages is:', message_freq[0][1], ' with ', message_freq[0][0], 'messages'
