from sqlitedict import SqliteDict
mydict = SqliteDict('./my_db.sqlite', autocommit=True)
# does not support tuples only strings
# mydict["key"] = 55

# try multiple keys at once
sample_dict  = {str(i):i for i in range(50)}
mydict.update(sample_dict)
mydict.close()