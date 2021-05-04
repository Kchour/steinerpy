# import os
# def file_handle(function):
#     """Function wrapper to extend file-handling capabilities
#         Assumes the first argument related to file behavior or the 'file_behavior' key exists


#     """
#     def wrapper(*args, **kwargs):
#         if args[0].file_behavior == "HALT":
#             if os.path.exists(args[0].cache_filename):
#                 raise FileExistsError('{} already exists!'.format(args[0].cache_filename))
#         elif args[0].file_behavior == "SKIP":
#             pass
#             return function(*args, **kwargs)
#         elif args[0].file_behavior == "RENAME":
#             cnt = 1
#             while True:
#                 temp = args[0].filename
#                 if os.path.exists(args[0].cache_filename)
#     return wrapper


# import os
# class File(object):
#     """Class to extend file-handling with behavior, to be used in conjunction with 
#         a context manager
    
#     Example
#         >>> from file_utils import File
#         >>> with File(file_name, method, behavior) as f
#                 now_do_something(f)
    
#     Args:
#         behavior (str): OVERWRITE-overwrites an existing file, SKIP-skip over an existing file,
#             HALT-stop when encountering an existing file, RENAME-append number to new file when file exists,
#             LOAD-load previous terminal data
#         method (str): 'rb'-to read, 'wb'-to write
#         file_name (str): name of the file to be opened (absolute path required)

#     """
#     def __init__(self, file_name, method, behavior="HALT"):
#         self.behavior = behavior
#         self.file_obj = None
#         if 'w' in method or '+' in method or 'a' in method:
#             if behavior == "HALT":
#                 if os.path.exists(file_name):
#                     raise FileExistsError('{} already exists!'.format(file_name))
            
#             elif behavior == "OVERWRITE":
#                 pass
#             elif behavior == "SKIP":
#                 if os.path.exists(file_name):
#                     pass
#                 else:
#                     SKIP = True
#             elif behavior == "RENAME":
#                 cnt = 1
#                 while True:
#                     temp = file_name
#                     if os.path.exists(temp):
#                         temp += str(cnt)
#                         cnt += 1
#                     else:
#                         file_name = temp
#                         break
#             else:
#                 SKIP = True

#             if not SKIP:
#                 self.file_obj = open(file_name, method)
#         else:
#             self.file_obj = open(file_name, method)

#     def __enter__(self):
#         return self.file_obj
#     def __exit__(self, type, value, traceback):
#         self.file_obj.close()

