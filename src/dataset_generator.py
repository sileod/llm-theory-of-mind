from Problem import *
import pandas as pd
import csv
import numpy as np
import re
from util import n_random_first_names, solve
from string import Template
from pandarallel import pandarallel

def get_agents_names(text):
    """
    Gets the names of the agents in the text
    :param text: The text where the names will be searched
    :return: A dictionary with the names as keys and the agent0, agent1, agent2, ... as values
    """
    # We could have used set() but we want to keep the order
    words = re.findall('[A-Z][a-z]+', text)

    # We create our own set to keep the order
    unique_names = {}

    # Count the number of agents
    agent = 0

    for name in words:
        # If the name is not in the set, we add it
        if name not in unique_names:
            # We bind the name to the agent number
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
        # If anonymize is True, we replace the names by the agent0, agent1, agent2, ...
        if anonymize:
            text = text.replace(name, names[name])
        # If anonymize is False, we replace the agent0, agent1, agent2, ... by the names
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
    'announcements': None,
    'n_announcements': 1,
    'hypothesis': None,
    'type': 'visual',
    'observation': '',
    'name': 'forehead'
}

arm_same_room = {
    'variables': None,
    'variables_template': '$agent arm is muddy',
    'agents': None,
    'law': Expression(Law('Top')),
    'matrix': np.ones((n_agents, n_agents)),
    'announcements': None,
    'n_announcements': 1,
    'hypothesis': None,
    'type': 'visual',
    'observation': 'all agents are in the same room.',
    'name': 'arm_same_room'
}

internal = {
    'variables': None,
    'variables_template': '$agent is thirsty',
    'agents': None,
    'law': Expression(Law('Top')),
    'matrix': np.eye(n_agents),
    'announcements': None,
    'n_announcements': 1,
    'hypothesis': None,
    'type': 'mental',
    'observation': '',
    'name': 'internal'
}

setups = [forehead, arm_same_room, internal]

Npb = 10
Nvariations = 100

if __name__ == '__main__':
    # Initialize the parallelization
    # It is used to solve the problems
    pandarallel.initialize(verbose=0, progress_bar=True, nb_workers=8)

    print('Dataset generation started !')
    print('Generating SMCDEL problems ...')

    # Count the number of generated problems
    generated_problems = 0

    # Create the dataframe
    df = pd.DataFrame(columns=['problem', 'premise', 'hypothesis'])

    # Generate the problems
    for _ in range(Npb):

        # Choose a random setup
        setup = np.random.choice(setups)
        
        # Generate the agents randomly
        setup['agents'] = n_random_first_names(n_agents)

        # Generate the variables randomly according to the setup
        setup['variables'] = [Var(Template(setup['variables_template']).substitute(agent=setup['agents'][i]), i+1) for i in range(len(setup['agents']))]

        # Setting the number of announcements
        setup['n_announcements'] = 2

        # Setting random announcements here to keep the same premises for all the variations
        setup['announcements'] = [Expression(Announcement(random.choice([random_expression(setup['variables'], 1), random_knowledge(setup['agents'], setup['variables'], 0)]))) for i in range(setup['n_announcements'])]

        # Checking if announcements cancel each other

        # Instanciating the problem
        pbcheck = Problem(**setup)
        # Setting the hypothesis to 0
        pbcheck.hypothesis = Expression(Var('0', 0))
        # Solving the problem
        label = solve(str(pbcheck).replace('VARS ', 'VARS 0,'))
        # As we never mention the 0 variable in premises, label should be 0
        # If the label is 1, there is likely a problem in the announcements
        # If so, we do not keep this problem
        while label == 1:
            pbcheck.announcements = [Expression(Announcement(random.choice([random_expression(setup['variables'], 1), random_knowledge(setup['agents'], setup['variables'], 0)]))) for i in range(setup['n_announcements'])]
            label = solve(str(pbcheck).replace('VARS ', 'VARS 0,'))

        # Create variations of the current problem with the same setup to get random hypotheses
        # By doing this, we can hope to get an hypothesis for each label
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
            df_pb = pd.DataFrame([[smcdel_pb, premise, hypothesis, setup['name']]], columns=['problem', 'premise', 'hypothesis', 'setup'])

            # Add the problem to the main dataframe
            df = pd.concat([df, df_pb], ignore_index=True)

            generated_problems += 1
        
            # Print the percentage of generated problems
            percentage = ((generated_problems / (Npb * Nvariations)) * 100)
            if percentage % 10 == 0:
                print(percentage, '% generated', sep='')
    
    print('SMCDEL problems generated !')

    print('Solving SMCDEL problems ...')

    # Solve the problems
    df['label'] = df['problem'].parallel_apply(solve)

    print('\nSMCDEL problems solved !')

    print('Preparing the dataset')

    # Get one problem for each premise/label couple
    one_of_each = pd.concat(
        [df.groupby(['premise', 'label'], group_keys=True).apply(lambda x: x.sample(1, random_state=i)) 
        for i in range(1000)]
    )

    one_of_each = one_of_each.drop_duplicates()

    one_of_each.rename(columns={'premise': 'prem'}, inplace=True)

    # Create the final dataframe
    final_df = pd.DataFrame(columns=['problem', 'premise', 'setup', 'true_hypothesis', 'false_hypothesis', 'hypothesis', 'label'])

    # For each premise, get the true and false hypotheses
    for premises, group in one_of_each.groupby('prem'):
        true_hyps = group[group['label'] == 1]['hypothesis']
        false_hyps = group[group['label'] == 0]['hypothesis']

        for i in range(min(len(true_hyps), len(false_hyps))):

            true_hyp = true_hyps.values[i]
            false_hyp = false_hyps.values[i]

            label = group['label'].values[i]
            problem = group['problem'].values[i]
            setup = group['setup'].values[i]

            if random.choice([True, False]):
                hypothesis = true_hyp
                label = 'entailment'
            else:
                hypothesis = false_hyp
                label = 'not_entailment'

            pb_df = pd.DataFrame([[
                problem,
                premises,
                setup,
                true_hyp,
                false_hyp,
                hypothesis,
                label
            ]], columns=['problem', 'premise', 'setup', 'true_hypothesis', 'false_hypothesis', 'hypothesis', 'label'])
            
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
    final_df.to_json('/Users/number/Dropbox/Applications/modlog/blablabla.jsonl', orient='records', lines=True)
