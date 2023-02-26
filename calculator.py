def add(a, b):
    return a+b
def subtract(a, b):
    return a-b
def multiply(a, b):
    return a*b
def divide(a, b):
    return a/b

print('Select Operation')
print('1. Add')
print('2. Subtract')
print('3. Multiply')
print('4. Divide')

while True:
    x = input("enter 1 to add, 2 to subtract, 3 to multiply, 4 to divide: ")
    if x in ('1', '2', '3', '4'):
        w = float(input("enter value: "))
        v = float(input("enter value: "))
        if x == '1':
            print(w, '+', v, '=', add(w, v))
        elif x == '2':
            print(w, '-', v, '=', subtract(w, v))
        elif x == '3':
            print(w, '*', v, '=', multiply(w, v))
        elif x == '4':
            print(w, '/', v, '=', divide(w, v))
        newcal = input("new calculation (yes/no): ")
        if newcal == "no":
            break
    else:
        print('invalid number')