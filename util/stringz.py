import re

number_regex = re.compile('(\d+)')
def split_out_numbers(lst):
    # credit: 
    # http://code.activestate.com/recipes/135435-sort-a-string-using-numeric-order/
    l = number_regex.split(lst)
    return [int(y) if y.isdigit() else y for y in l]

