import exrex
import pandas as pd
import dropbox

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

def generate_announcement_regex(vars, agents, n_announcements, statement_size):
    agents_regex = '|'.join(agents)
    vars_regex   = '|'.join(map(str, vars))

    agents_knowledge = f'(~ )?\( ({agents_regex}) knows whether ({vars_regex}) \)'
    statement = f'(~ )?\( ({vars_regex})(( & | \| )({vars_regex})){{{statement_size-1}}} \)'
    reg = f'(\[ ! (({statement})|({agents_knowledge})) \]){{{n_announcements}}} (({statement})|({agents_knowledge}))'

    return reg

def generate_problem_regex(n_vars, n_agents, n_observations):
    vars = generate_vars(n_vars)
    agents = generate_agents(n_agents)

    law_regex = generate_law()

    observations_regex = generate_observations_regex(vars, agents, n_observations)

    announcements_regex = generate_announcement_regex(vars, agents, n_announcements=1, statement_size=3)

    problem_regex = f"VARS {','.join(map(str, vars))} LAW {law_regex} OBS {observations_regex} VALID\? {announcements_regex}\n"

    return problem_regex


with open('./token.txt', 'r') as f:
    token = f.read()

def to_dropbox(dataframe, path, token):
    dbx = dropbox.Dropbox(token)

    df_string = dataframe.to_csv(index=False)
    db_bytes = bytes(df_string, 'utf8')
    
    dbx.files_upload(
        f=db_bytes,
        path=path,
        mode=dropbox.files.WriteMode.overwrite
    )


if __name__ == '__main__':
    problem_regex = generate_problem_regex(n_vars=3, n_agents=3, n_observations=2)

    with open('./problems.txt', 'w') as f:
        df = pd.DataFrame(columns=['problem'])
        for _ in range(20000):
            pb = exrex.getone(problem_regex)
            df_pb = pd.DataFrame([pb], columns=['problem'])

            df = pd.concat([df, df_pb], ignore_index=True)
        df.to_csv(f, index=False)

    to_dropbox(df, '/Applications/modlog/problems2.csv', token)



