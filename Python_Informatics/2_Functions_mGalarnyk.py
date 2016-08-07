'''
Michael Galarnyk

1. Define a function named to_number(str) that takes a string as a parameter, converts it to an int value, and returns that int value.
2. Define a function named add_two(n1, n2) that takes two ints as parameters, sums them, and then returns that int sum value.
3. Define a function named cube(n) that takes numeric value as a parameter, cubes that value, and then returns that resulting numeric value.
4. Use the above functions in one statement to take two string literals, convert them to ints, add them together, cube the result, and print the cubed value.

'''

def to_number(string):
    new_int = int(string)
    return new_int

def add_two(n1,n2):
    summation = n1 + n2
    return summation

def cube(n): 
    cubed = n**3
    return cubed

print cube(add_two(to_number('6'),to_number('5')))
