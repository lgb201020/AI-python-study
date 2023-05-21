"""list의 끝이 stack의 last인 경우"""
class Stack_case_1:
    def __init__(self):
        self.items = []
    
    def is_empty(self):
        return self.items == []
    '''비어있으면 TRUE 값을 출력'''
    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()
    
    def peek(self):
        return self.items[len(self.items)-1]

    def size(self):
        return len(self.items)
    
    def pop_all(self):
        return self.items.clear()


"""list의 맨 앞이 stack의 last인 경우"""
class Stack_case_2:
    def __init__(self):
        self.items = []
    """init은 생성자이다. 즉 __***__이렇게 함수를 정의 하면 생성자를 정의 하는 것이다."""
    def is_empty(self):
        return self.items == []
    '''비어있으면 TRUE 값을 출력'''
    def push(self, item):
        self.items.insert(0, item)

    def pop(self):
        return self.items.pop(0)
    
    def peek(self):
        return self.items[len(self.items)-1]

    def size(self):
        return len(self.items)
    
    def pop_all(self):
        return self.items.clear()
         

def reverse(str):
    st = Stack_case_2() 
    '''객체생성 and 생성자로 인해 리스트도 자동 생성'''
    for i in range(len(str)):
        st.push(str[i])

    out = ""
    while not st.is_empty():
        out += st.pop()
    
    return out

input = "test seq 12345"
answer = reverse(input)

print("input string: ", input)
print("reversed string: ", answer)
