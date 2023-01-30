import exrex

with open('./names.txt', 'r') as f:
    names = f.read().split('\n')

def generate_vars(n_vars):
    return list(range(1, n_vars+1))

def generate_agents(n_agents):
    return names[:n_agents]

def generate_law():
    return exrex.getone('Top')

def generate_observations_regex(vars, agents, n_observations):
    vars_regex   = '|'.join(map(str, vars))
    agents_regex = '|'.join(agents)
    return '((' + agents_regex + '):('+ vars_regex +')(,('+ vars_regex + ')){' + str(n_observations-1) + '} ){' + str(len(agents)) + '}'

def generate_problem_regex(n_vars, n_agents, n_observations):
    vars = generate_vars(n_vars)
    agents = generate_agents(n_agents)

    law_regex = generate_law()

    observations_regex = generate_observations_regex(vars, agents, n_observations)

    problem_regex = f"VARS {','.join(map(str, vars))} LAW {law_regex} OBS {observations_regex}\n"

    return problem_regex

with open('./problems.txt', 'w') as f:
    for pb in list(exrex.generate(generate_problem_regex(3, 3, 2))):
        f.write(pb)

print(len(list(exrex.generate(generate_problem_regex(3, 3, 2)))))
