def myams(s):
    a = []
    for i in s:
        a.append(int(i)**3)
    return (sum(a) == int(s))


print("please enter the input:")
print(myams(str(input())))