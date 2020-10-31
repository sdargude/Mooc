
a = ["one","two","three", "four", "five", "six"]
windows = 2
size = len(a)
print(a)
for idx, w in enumerate(a):
    left = a[max(0,idx-windows):idx]
    right = a[idx+1:min(size,idx+windows+1)]
    print(idx,a[idx],"----->",left+right)
    print("----------------------------------")
