'''알파벳 대소문자로 된 단어가 주어지면, 이 단어에서 가장 많이 사용된 알파벳이 무엇인지 알아내는 프로그램(알고리즘)을 작성하시오. 
(대문자와 소문자를 구분하지 않는다. 첫째 줄에 이 단어에서 가장 많이 사용된 알파벳을 대문자로 출력한다. 
단, 가장 많이 사용된 알파벳이 여러 개 존재하는 경우에는 ?를 출력한다.)'''

'''
1. 단어 입력 함수로 단어를 입력한다.
2. lower()함수를 사용해서 문자열을 소문자로 바꿔준다.
3. set형태로 바꿔준다.
4. for문으로 리스트에서 문자를 하나씩 뽑아낸다.
5. if문으로 set안에 문자를 list 이름으로 하는 배열을 만들고 index 0 에 리스트 이름을 넣어준다. 그리고 append()함수로 1을 넣어준다.
6. 이후 완료후 리스트 길이중 가장 긴 리스트를 출력


word = input("단어를 입력하세요 : ")
word_lower = word.lower()
word_lower_set = set(word_lower)
word_lower_set_to_list=list(word_lower_set)

a = [[word_lower_set_to_list[i]]*1 for i in range(len(word_lower_set))]

for i in range(len(word_lower)):
    w = word_lower[i]
    for j in range(len(word_lower_set_to_list)):
        if  w == a[j][0]:
            a[i].append(i)

print(a)

위에 처럼 그냥 써도 상관없지만 행렬을 만드는 함수를 정의하자'''

'''단어를 문자별로 분류해 놓은 행렬'''

def matrix(w):
    '''nrow는 단어가 갖고있는 문자 종류 개수이다.'''
    rows = []
    m = []
    w_list = list(w)
    nrow = len(set(w_list))
    w_set_list = list(set(w_list))

    for i in range(nrow):
        rows = []
        rows.append(w_set_list[i])
        m.append(rows)
    '''반복하는데 비어있는 1차 리스트를 만들고 거기에 단어에 들어가는 문자 종류를 하나 넣은뒤 그 행을 다시 m 리스트에 넣어 
    각 행의 첫번째 요소가 단어에 들어가는 문자 종류가 되는 행렬을 만듬'''
    return m

def count_character(w):
    '''단어에 있는 문자수 count'''
    w_character = matrix(w)
    w_list = list(w) 

    for i in range(len(w_list)):
        for j in range(len(w_character)):
            if w_list[i] == w_character[j][0]:
                w_character[j].append(1)
    return w_character

def max_size_row(word):
    """단어를 받아 2차 배열에서 크기가 가장 큰 행을 선택, index 0 element를 출력 단 아직 가장 큰 값이 중복되는 경우는 고려 안함"""
    m = count_character(word)
    list =[]
    for i in range(len(m)):
        size = len(m[i])
        list.append(size)
    '''
    최댓값 찾는 알고리즘을 직접 구현한 것
    tmp = list[0]
    for j in range(1,len(list)):
        if list[j] > tmp:
            tmp = list[j]
    '''
    if len(list) == len(set(list)):
        most_frequent_character = m[list.index(max(list))][0]
        return most_frequent_character
    elif len(list) > len(set(list)):
        return "?"


letteral = input("단어를 입력하세요 : ")
print(max_size_row(letteral))