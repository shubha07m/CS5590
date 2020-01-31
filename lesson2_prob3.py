def wordcount(t):
    mapdict = {}
    for line in t:
        for word in line.split(" "):
            if "\n" in word:
                word = word[:-1]
            if word not in mapdict:
                mapdict[word] = 1
            else:
                mapdict[word] += 1

    return mapdict


f = open("abc.txt", "r")
fnew = open("newfile.txt","w")
fnew.write(str(wordcount(f)))
fnew.close()
fnewer = open("newfile.txt", "r")
print(fnewer.read())
fnewer.close()
f.close()