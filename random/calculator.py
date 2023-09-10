def add(a, b):
    return a+b
def subtract(a, b):
    return a-b
def multiply(a, b):
    return a*b
def divide(a, b):
    return a/b
def x2(a):
    return a ** 2

print('Select Operation')
print('1. Add')
print('2. Subtract')
print('3. Multiply')
print('4. Divide')
print('5. Power 2')

while True:
    y = input('enter 1 for basic operations, 2 for others:')
    if y == '1':
        x = input("enter 1 to add, 2 to subtract, 3 to multiply, 4 to divide, 5 to power by 2: ")
        if x in ('1', '2', '3', '4', '5'):
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
    elif y == '2':
        z = float(input("enter single value for operation: "))
        print(z, x2(z))
        newcal2 = input('new calculation (yes/no)')
        if newcal2 == 'no':
            break
    else:
        print('invalid number')