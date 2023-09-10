name = 'leinad'

def get_in():
    return f'this is my test module have fun working with it'


class get:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
    def __str__(self):
        return f'my name is {self.name} and i am {self.age} years old'