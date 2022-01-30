#  1e8 is too large, will freeze system
# try:
#     test = {(1, x): 1 for x in range(int(1e8))}
# except MemoryError as e:
#     raise e

# the following is approximately 7 gigs of ram

print("1")
test1 = {(1, x): 1 for x in range(int(1e7))}
print("2")
test2 = {(1, x): 1 for x in range(int(1e7))}
print("3")
test3 = {(1, x): 1 for x in range(int(1e7))}
print("4")
test4 = {(1, x): 1 for x in range(int(1e7))}
print("5")
test5 = {(1, x): 1 for x in range(int(1e7))}
print("6")
test6 = {(1, x): 1 for x in range(int(1e7))}