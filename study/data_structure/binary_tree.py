import numpy as np
table =[('A', '.-'),    ('B', '-...'),  ('C', '-.-.'),  ('D', '-..'),
        ('E', '.'),     ('F', '..-.'),  ('G', '--.'),   ('H', '....'),
        ('I', '..'),    ('J', '.---'),  ('K', '-.-'),   ('L', '.-..'),
        ('M', '--'),    ('N', '-.'),    ('O', '---'),   ('P', '.--.'),
        ('Q', '--.-'),  ('R', '.-.'),   ('S', '...'),   ('T', '-'),
        ('U', '..-'),   ('V', '...-'),  ('W', '.--'),   ('X', '-..-'),
        ('Y', '-.--'),  ('Z', '--..') ]
table = np.asarray([sublist for sublist in table])



'''tree:[["a",".-"][][]]
function: insert함수가 root를 n번째 node로 가졌을때 n+1번째 node를 삽입하고 n+2번째 node로 가는 branch를 생성
'''

def insert(root,newbranch):
    t = root.pop(1)
    if len(t) > 1:
        root.insert(1,[newbranch,t,[]])
    else:
        root.insert(1,[newbranch,[],[]])
    return root



class TNode:								
    def __init__ (self, data, left, right):	
        self.data = data 					
        self.left = left					
        self.right = right

def make_mos_tree():
    """모스부호에 맞는 비어있는 bianry tree를 만드는 함수: ??? 이해가...."""
    root = TNode(None,None,None)
    for tp in table:
        code = table[1]
        node = root
        for c in code:
            if c == ".":
                if node.left == None:
                    node.left = TNode(None,None,None)
                node = node.left
                """일단 비어있는 트리를 만듬"""
            elif c == "-":
                if node.right == None:
                    node.right = TNode(None,None,None)
                node = node.right

        node.data = tp[0]
        return root

def decode(root, code):
    node = root
    for c in code:
        if c == ".": node = node.left
        elif c =="-": node = node.right
    return node.data

def encode(ch):
    idx = ord(ch) - ord("A")
    return table[idx][1]




"""
def make_binarytree(tree_name,table):
    make_empty = [[],[],[]]
    tree_name = make_empty[0].append(tree_name)
    tree_name = [[tree_name][][]] : 첫번째 2번째 node를 빈 리스트로 만듬 - defaultset
    .과-를 구분
    """
    


