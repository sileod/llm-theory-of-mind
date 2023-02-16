import censusname

def random_first_name():
    return censusname.generate(nameformat='{given}')

def n_random_first_names(n):
    names = []
    for _ in range(n):
        name = random_first_name()
        while name in names:
            name = random_first_name()
        names.append(name)
    return names