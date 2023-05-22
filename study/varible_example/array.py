import numpy as np
table =[('A', '.-'),    ('B', '-...'),  ('C', '-.-.'),  ('D', '-..'),
        ('E', '.'),     ('F', '..-.'),  ('G', '--.'),   ('H', '....'),
        ('I', '..'),    ('J', '.---'),  ('K', '-.-'),   ('L', '.-..'),
        ('M', '--'),    ('N', '-.'),    ('O', '---'),   ('P', '.--.'),
        ('Q', '--.-'),  ('R', '.-.'),   ('S', '...'),   ('T', '-'),
        ('U', '..-'),   ('V', '...-'),  ('W', '.--'),   ('X', '-..-'),
        ('Y', '-.--'),  ('Z', '--..') ]
table_to_list = np.asarray([sublist for sublist in table])
print(table_to_list[0][0][1])



'''
for i in range(len(table)):
    list(table[i])
    list(table[i][1])
    table 의 모든 str,tuple을 list형태의 자료형으로 만듬"""

print(table)
'''
'''
a = [[]*1 for i in range()]
'''

