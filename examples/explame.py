v = locals()
lst1 = range(10)

for i in lst1:
    v['A'+ str(i+1)] = [1,2,3]

print(A1)
d = v.copy()
# for var in d:
#     if "A" in var and "__" not in var:
#         print(eval(var)[2])