def type1(a):
    b = []
    for i in a:
        b.append(int(i) * 0.453592)
    return b


def type2(a):
    return [i * .453592 for i in a]


L1 = [150, 155, 145, 148]
print(type1(L1))
print(type2(L1))