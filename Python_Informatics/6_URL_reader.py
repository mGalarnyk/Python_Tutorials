'''
Michael Galarnyk

1. Rename the socket1.py program from our textbook to URL_reader.py.
2. Modify the URL_reader.py program to use urllib instead of a socket.
3. Add code that prompts the user for the URL so it can read any web page.
4. Add error checking using try and except to handle the condition where the user enters an
improperly formatted or non-existent URL.
5. Count the number of characters received and stop displaying any text after it has shown 3000
characters.
6. Continue to retrieve the entire document, count the total number of characters, and display the
total number of characters.
'''
# Wasnt sure what exactly the assignment wanted, but this seems to work.

import urllib
import re

# https://www.gutenberg.org/files/1342/1342-h/1342-h.htm#link2HCH0034

try: 
    url = raw_input('Enter - ');
    fhand = urllib.urlopen(url);
    count = 0; # only characters upper and lowercase a-z.
    adjusted_3000 = [];
    while count < 3000:
        new_char = fhand.read(1);
        adjusted_3000 = adjusted_3000 + re.findall('[A-Za-z]', new_char)
        count += 1; 
    print 'first 3000 thousand characters are: ', adjusted_3000;
    
    new_chars = fhand.read()
    entire_doc = adjusted_3000 + re.findall('[A-Za-z]', new_chars)
    print 'The length of the entire document is: ', len(entire_doc)    
except: 
    print "Bad link"


