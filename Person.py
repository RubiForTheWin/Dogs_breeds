class Person(object):
    def __init__(self, Fname, Mname, age):
        self.Fname = Fname
        self.Mname = Mname
        self.age = age
        self.__age = age
        Person.counter += 1

        def get_age(self):
            return self.__age

        def set_age(self, age):
            if age >= 0:
                self.__age = age
            else:
                print("i.e")

        def get_Fname(self):
            return self.__Fname

        def set_Fname(self, Fname):
            self.Fname=Fname

            def get_Lname(self, Mname):
                return self.__Mname

            def set_Lname(self, Lname):
                self.Lname = Lname


p1 = Person("uzi", "chafuzi", 12)
p2 = Person("mozi", "makfuzi", 15)

