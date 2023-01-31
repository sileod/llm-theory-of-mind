import exrex
import pandas as pd
import sys
from util import to_dropbox, get_url_of_file

# Read the names from the file
with open('./names.txt', 'r') as f:
    names = f.read().split('\n')

# Read the token from the file
with open('./token.txt', 'r') as f:
    token = f.read()

def generate_vars(n_vars):
    """
    Generate a list of variables from 1 to n_vars
    :param n_vars: the number of variables
    :return: a list of variables
    """
    return list(range(1, n_vars+1))

def generate_agents(n_agents):
    """
    Return the first n_agents names
    :param n_agents: the number of agents
    :return: a list of names
    """
    return names[:n_agents]

def generate_law():
    """
    """
    return exrex.getone('Top')

def generate_observations_regex(vars, agents, n_observations):
    """
    Generates a regex that matches the declaration of observations
    :param vars: the list of variables
    :param agents: the list of agents
    :param n_observations: the number of observations
    :return: a regex
    """
    vars_regex   = '|'.join(map(str, vars))
    agents_regex = '|'.join(agents)
    return '((' + agents_regex + '):('+ vars_regex +')(,('+ vars_regex + ')){' + str(n_observations-1) + '} ){' + str(len(agents)) + '}'

def generate_announcement_regex(vars, agents, n_announcements, statement_size):
    """
    Generates a regex that matches the declaration of announcements
    :param vars: the list of variables
    :param agents: the list of agents
    :param n_announcements: the number of announcements
    :param statement_size: the size of the statements
    """
    agents_regex = '|'.join(agents)
    vars_regex   = '|'.join(map(str, vars))

    agents_knowledge = f'(~ )?\( ({agents_regex}) knows whether ({vars_regex}) \)'
    statement = f'(~ )?\( ({vars_regex})(( & | \| )({vars_regex})){{{statement_size-1}}} \)'
    reg = f'(\[ ! (({statement})|({agents_knowledge})) \]){{{n_announcements}}} (({statement})|({agents_knowledge}))'

    return reg

def generate_problem_regex(n_vars, n_agents, n_observations, n_announcements, statement_size):
    """
    Generates a regex that matches an epistemic problem
    :param n_vars: the number of variables
    :param n_agents: the number of agents
    :param n_observations: the number of observations per agent
    :param n_announcements: the number of announcements
    :param statement_size: the size of the statements
    :return: a regex
    """
    vars = generate_vars(n_vars)
    agents = generate_agents(n_agents)

    law_regex = generate_law()

    observations_regex = generate_observations_regex(vars, agents, n_observations)

    announcements_regex = generate_announcement_regex(vars, agents, n_announcements, statement_size)

    problem_regex = f"VARS {','.join(map(str, vars))} LAW {law_regex} OBS {observations_regex} VALID\? {announcements_regex}"

    return problem_regex

def generate_problems_to_csv(path, result_path, n_problems, n_vars=3, n_agents=3, n_observations=2, n_announcements=1, statement_size=3):
    problem_regex = generate_problem_regex(n_vars, n_agents, n_observations, n_announcements, statement_size)

    print(f'Generating {n_problems} problems...')

    nb_generated = 0

    df = pd.DataFrame(columns=['problem'])
    for _ in range(n_problems):
        pb = exrex.getone(problem_regex)
        df_pb = pd.DataFrame([pb], columns=['problem'])

        df = pd.concat([df, df_pb], ignore_index=True)

        # Print the progress
        nb_generated += 1
        if nb_generated % (n_problems//10) == 0:
            print(f'{nb_generated/n_problems*100} % generated')

    print('Generating done')

    print('Uploading to DropBox...')

    # Upload the file to Dropbox
    to_dropbox(df, f'/{result_path}', token)

    print('Upload done')
    
    # Get url of the uploaded file
    url = get_url_of_file(result_path, token)

    print(f'{result_path} uploaded to DropBox : {url}')

    return url

if __name__ == '__main__':
    path = sys.argv[1]
    result_path = sys.argv[2]
    n_problems = int(sys.argv[3])

    generate_problems_to_csv(path, result_path, n_problems)



