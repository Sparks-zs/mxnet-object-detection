num = 0
class test1:
    def __init__(self):
        global num
        self.num = num

    def run(self):
        self.num = 1
        print(num)

class test2:
    def __init__(self):
        num = 2

    def run(self):
        print(num)


t1 = test1()
t2 = test2()
t1.run()
t2.run()
