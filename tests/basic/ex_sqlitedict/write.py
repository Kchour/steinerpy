from sqlitedict import SqliteDict
from timeit import default_timer as timer
# flag ='r' is read only
mydict = SqliteDict('./my_db.sqlite', journal_mode="OFF", flag='r')
# does not support tuples only strings
# mydict["key"] = 55


num_of_keys = int(500e6)
# try multiple keys at once
t1 = timer()
for i in range(num_of_keys):
    if i % 1e6 == 0:
        print(i)
        mydict.commit()
    mydict[i] = i
mydict.commit()
mydict.close()
t2 = timer()
print("wrote {} in {}".format(num_of_keys, t2-t1))