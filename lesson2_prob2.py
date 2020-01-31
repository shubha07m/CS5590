def string_alternative(t):
    return t[::2]

def mystring(s):
    a = []
    for i in range(0, len(s) - 1, 2):
        a.append(s[i])
    return "".join(a)


print("please enter the string you want:")
this = input()
print(string_alternative(this))
print(mystring(this))
