class Robot:
    def __init__(self, name, color, weight) -> None:
        self.name = name
        self.color = color
        self.weight = weight
    
    def __str__(self) -> str:
        return f"my name is {self.name}, and i am color {self.color}, and i weigh {self.weight}"
    


r1 = Robot("dan", "green", 80)
print(r1)

r2 = Robot('enoch', 'red', 69)
print(r2)