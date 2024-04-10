A = {'a':1,'b':2}

def change(dic):
    dic['a'] = 3
    return dic
B = change(A)
print(B)
print(A)