__author__ = 'tomerp'

get_Prepostion_List = [line.rstrip('\n') for line in open('PrepositionsList')]
print(get_Prepostion_List.__contains__("in"))