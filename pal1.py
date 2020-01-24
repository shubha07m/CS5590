def mypal(s):
    return (s == s[::-1])

print("please enter the input:")
x = str(input())
print(mypal(x))