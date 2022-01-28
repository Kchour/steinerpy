from sqlitedict import SqliteDict
mydict = SqliteDict('./my_db.sqlite', autocommit=True)
# print(mydict["key"])
for k,v in mydict.items():
    print(k,v)
mydict.close()