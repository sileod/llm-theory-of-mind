from Problem import *
import pandas as pd
import csv
import numpy as np
import re
from name_generation import n_random_first_names
from string import Template
from solver import solve
from pandarallel import pandarallel
from util import to_dropbox, get_url_of_file

def get_agents_names(text):
    """
    Gets the names of the agents in the text
    :param text: The text where the names will be searched
    :return: A dictionary with the names as keys and the agent0, agent1, agent2, ... as values
    """
    # We could have used set() but we want to keep the order
    words = re.findall('[A-Z][a-z]+', text)
    unique_names = {}
    agent = 0
    for name in words:
        if name not in unique_names:
            unique_names[name] = f'agent{agent}'
            agent += 1
    return unique_names

def replace_names(text, names, anonymize=True):
    """
    Replaces the names in the text by agent0, agent1, agent2, ... when anonymize is True
    Replaces the agent0, agent1, agent2, ... by the names in the text when anonymize is False
    :param text: The text where the names will be replaced
    :param names: A dictionary with the names as keys and the agent0, agent1, agent2, ... as values
    :param anonymize: If True, the names will be replaced by agent0, agent1, agent2, ..., if False, the agent0, agent1, agent2, ... will be replaced by the names
    :return: The text with the names replaced
    """
    for name in names:
        if anonymize:
            text = text.replace(name, names[name])
        else:
            text = text.replace(names[name], name)
    return text

n_agents = 3

# Define the setups
forehead = {
    'variables': None,
    'variables_template': '$agent is muddy',
    'agents': None,
    'law': Expression(Law('Top')),
    'matrix': np.ones((n_agents, n_agents)) - np.eye(n_agents),
    'announcements': None,#[Expression(Announcement(random.choice([random_expression(variables, 1), Knowledge(random.choice(agents), random_expression(variables, 0))]))) for i in range(1)],
    'n_announcements': 1,
    'hypothesis': None,
    'type': 'visual',
    'observation': '',
}

arm_same_room = {
    'variables': None,
    'variables_template': '$agent arm is muddy',
    'agents': None,
    'law': Expression(Law('Top')),
    'matrix': np.ones((n_agents, n_agents)),
    'announcements': None,#[Expression(Announcement(random.choice([random_expression(variables, 1), Knowledge(random.choice(agents), random_expression(variables, 0))]))) for i in range(1)],
    'n_announcements': 1,
    'hypothesis': None,
    'type': 'visual',
    'observation': 'all agents are in the same room.'
}

internal = {
    'variables': None,
    'variables_template': '$agent is thirsty',
    'agents': None,
    'law': Expression(Law('Top')),
    'matrix': np.eye(n_agents),
    'announcements': None,#[Expression(Announcement(random.choice([random_expression(variables, 1), Knowledge(random.choice(agents), random_expression(variables, 0))]))) for i in range(1)],
    'n_announcements': 1,
    'hypothesis': None,
    'type': 'mental',
    'observation': ''
}

setups = [forehead, arm_same_room, internal]

Npb = 2000
Nvariations = 5

if __name__ == '__main__':
    pandarallel.initialize(verbose=0, progress_bar=True)

    print('Dataset generation started')

    print('Generating problems')

    generated_problems = 0

    # Create the dataframe
    df = pd.DataFrame(columns=['problem', 'premise', 'hypothesis'])
    for _ in range(Npb):

        # Choose a random setup
        setup = np.random.choice(setups)
        
        setup['agents'] = n_random_first_names(n_agents)
        setup['variables'] = [Var(Template(setup['variables_template']).substitute(agent=setup['agents'][i]), i) for i in range(len(setup['agents']))]

        # Create 5 problems with the same setup to get random hypotheses
        for _ in range(Nvariations):

            # Creating the problem
            pb = Problem(**setup)

            # Get the problem in smcdel format
            smcdel_pb = str(pb)

            # Switch to natural language
            pb.change_format('natural')

            # Get the problem in natural language
            premise  = pb.base_observation + str(pb.law) + pb.observations_to_str() + pb.announcements_to_str()

            # Get the hypothesis
            hypothesis = str(pb.hypothesis)

            # Create a dataframe with the problem, the premise and the hypothesis
            df_pb = pd.DataFrame([[smcdel_pb, premise, hypothesis]], columns=['problem', 'premise', 'hypothesis'])

            # Add the problem to the main dataframe
            df = pd.concat([df, df_pb], ignore_index=True)

            generated_problems += 1
        
            percentage = ((generated_problems / (Npb * Nvariations)) * 100)
            if percentage % 10 == 0:
                print(percentage, '% generated')
    
    print('Problems generated')

    print('Solving problems')

    df['label'] = df['problem'].parallel_apply(solve)

    print('Problems solved')

    print('Preparing the dataset')

    one_of_each = df.groupby(['premise', 'label']).apply(lambda x: x.sample(1, random_state=42))
    one_of_each.rename(columns={'premise': 'prem'}, inplace=True)

    final_df = pd.DataFrame(columns=['problem', 'premise', 'true_hypothesis', 'false_hypothesis', 'hypothesis', 'label'])

    for premises, group in one_of_each.groupby('prem'):
        true_hyps = group[group['label'] == 1]['hypothesis']
        false_hyps = group[group['label'] == 0]['hypothesis']

        if len(true_hyps) == 0 or len(false_hyps) == 0:
            continue

        true_hyp = true_hyps.values[0]
        false_hyp = false_hyps.values[0]

        label = group['label'].values[0]
        problem = group['problem'].values[0]

        if random.choice([True, False]):
            hypothesis = true_hyp
            label = 'entailment'
        else:
            hypothesis = false_hyp
            label = 'not_entailment'

        pb_df = pd.DataFrame([[problem, premises, true_hyp, false_hyp, hypothesis, label]], columns=['problem', 'premise', 'true_hypothesis', 'false_hypothesis', 'hypothesis', 'label'])
        
        final_df = pd.concat([final_df, pb_df], ignore_index=True)

    print('Dataset prepared')

    # Anonymize the names
    # We need to do this because we want to remove as many duplicates as possible
    final_df['names'] = final_df['premise'].apply(get_agents_names)
    final_df['premise'] = final_df.apply(lambda x: replace_names(x['premise'], x['names'], anonymize=True), axis=1)
    final_df['hypothesis'] = final_df.apply(lambda x: replace_names(x['hypothesis'], x['names'], anonymize=True), axis=1)
    # Remove duplicates
    final_df = final_df.drop_duplicates(subset=['premise', 'hypothesis']).reset_index(drop=True)

    # De-anonymize the names
    final_df['premise'] = final_df.apply(lambda x: replace_names(x['premise'], x['names'], anonymize=False), axis=1)
    final_df['hypothesis'] = final_df.apply(lambda x: replace_names(x['hypothesis'], x['names'], anonymize=False), axis=1)

    # Remove the names column
    final_df = final_df.drop(columns=['names'])

    # Save the dataframe to a jsonl file
    final_df.to_json('/Users/number/Dropbox/Applications/modlog/dataset.jsonl', orient='records', lines=True)

    # to_dropbox(final_df, '/dataset.jsonl')

    # url = get_url_of_file('/dataset.jsonl')

    # print(url)
