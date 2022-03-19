class Human:
    __name = None
    __age = 0
    __sex = None

    def __init__(self, name, age, sex):
        self.__name = name
        self.__age = age
        self.__sex = sex

    def get_name(self):
        return self.__name

    def get_age(self):
        return self.__age

    def get_sex(self):
        return self.__sex


def some_func(arr):
    arr.sort()
    for i in arr[::-1]:
        print(i)
    return arr[1]


if __name__ == '__main__':
    print("hey")
    my_set = {5, 4, 3, 2}
    my_set.add(6)
    for i in my_set:
        print(i)
    hm = Human("Guy", 24, "male")
    print(hm.get_age())
