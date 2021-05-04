# Python code to demonstrate  
# how parent constructors are called.  
    
# parent class  
class Person( object ):      
    
        # __init__ is known as the constructor           
        def __init__(self, name, idnumber):     
                self.name = name  
                self.idnumber = idnumber  
                  
        def display(self):  
                print(self.name)  
                print(self.idnumber)  
    
# child class  
class Employee( Person ):             
        def __init__(self, name, idnumber, salary):  
                self.salary = salary  
    
                # invoking the constructor of  
                # the parent class   
                Person.__init__(self, name, idnumber)   
          
        def show(self): 
            print(self.salary) 

# creation of an object 
# variable or an instance  
a = Employee('Rahul', 886012, 30000000)      
    
# calling a function of the 
# class Person using Employee's 
# class instance  
a.display() 
a.show()  
print(a.name)