from Problem import *
import pandas as pd
import numpy as np
import random
import re
from util import n_random_first_names, solve, class_to_dict
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
        if anonymize:
            text = text.replace(name, names[name])
        else:
            text = text.replace(names[name], name)
    return text

n_agents = 3

class Setup:
    variables=None
    agents=None
    announcements= None
    n_announcements= 1
    hypothesis= None
    type=None
    observation=None
    law=None#Expression(Law('Top'))
    
class forehead(Setup):
    observation='Everyone is visible to others. '
    variables_template="$agent's forehead  is muddy"
    matrix= np.ones((n_agents, n_agents)) - np.eye(n_agents)
    type='visual'
    
class arm(forehead):
    variables_template="$agent's arm is muddy"
    matrix= np.ones((n_agents, n_agents)) 

class internal(Setup):
    observation='Everyone is visible to others. '
    variables_template='$agent is thirsty'
    matrix= np.eye(n_agents) 
    type='mental'

setups = [forehead, arm, internal]
setups = [class_to_dict(x) for x in setups]


def generate(Npb = 10, Nvariations = 100):
    # Initialize the parallelization
    # It is used to solve the problems
    pandarallel.initialize(verbose=0, progress_bar=False, nb_workers=8)

    # Create the dataframe
    df = pd.DataFrame(columns=['problem', 'premise', 'hypothesis'])

    # Generate the problems
    for _ in range(Npb):

        setup = np.random.choice(setups)
        setup['agents'] = n_random_first_names(n_agents)
        setup['variables'] = [Var(Template(setup['variables_template']).substitute(agent=setup['agents'][i]), i+1) for i in range(len(setup['agents']))]
        base_pb = Problem(**setup)
        setup['n_announcements'] = 2
        setup['announcements'] = [Expression(Announcement(random.choice([
            base_pb.random_expression(1), base_pb.random_knowledge(0)])))
            for i in range(setup['n_announcements'])
        ]

        # Instanciating the problem
        pbcheck = Problem(**setup)
        pbcheck.hypothesis = Expression(Var('0', 0))
        label = solve(str(pbcheck).replace('VARS ', 'VARS 0,'))
        # As we never mention the 0 variable in premises, label should be 0
        # If the label is 1, there is likely a problem in the announcements
        # If so, we generate new announcements until there is no problem
        while label == 1:
            pbcheck.announcements = [Expression(Announcement(random.choice([
                base_pb.random_expression(1), 
                base_pb.random_knowledge(0)]))) for i in range(setup['n_announcements'])]
            label = solve(str(pbcheck).replace('VARS ', 'VARS 0,'))

        # Create variations of the current problem with the same setup to get random hypotheses
        # By doing this, we can hope to get an hypothesis for each label
        for _ in range(Nvariations):

            # Creating the problem
            pb = Problem(**setup)

            # Get the problem in smcdel format
            smcdel_pb = str(pb)
            pb.change_format('natural')
            premise  = pb.observations_to_str() + str(pb.law) + pb.announcements_to_str()
            premise=premise.replace('.', '. ').replace('  ',' ')
            hypothesis = str(pb.hypothesis)

            # Create a dataframe with the problem, the premise and the hypothesis
            df_pb = pd.DataFrame([[smcdel_pb, premise, hypothesis, setup['name']]], columns=['problem', 'premise', 'hypothesis', 'setup'])
            df = pd.concat([df, df_pb], ignore_index=True)
        
    df['label'] = df['problem'].parallel_apply(solve)

    one_of_each = pd.concat(
        [df.groupby(['premise', 'label'], group_keys=True).apply(lambda x: x.sample(1, random_state=i)) 
        for i in range(1000)]
    )

    one_of_each = one_of_each.drop_duplicates()

    one_of_each.rename(columns={'premise': 'prem'}, inplace=True)

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

    # Anonymize the names and deduplicate
    final_df['names'] = final_df['premise'].apply(get_agents_names)
    final_df['premise'] = final_df.apply(lambda x: replace_names(x['premise'], x['names'], anonymize=True), axis=1)
    final_df['hypothesis'] = final_df.apply(lambda x: replace_names(x['hypothesis'], x['names'], anonymize=True), axis=1)
    final_df = final_df.drop_duplicates(subset=['premise', 'hypothesis']).reset_index(drop=True)

    # De-anonymize the names
    final_df['premise'] = final_df.apply(lambda x: replace_names(x['premise'], x['names'], anonymize=False), axis=1)
    final_df['hypothesis'] = final_df.apply(lambda x: replace_names(x['hypothesis'], x['names'], anonymize=False), axis=1)
    final_df = final_df.drop(columns=['names'])

    final_df.to_json('epistemic-logic-reasoning.jsonl', orient='records', lines=True)
    return final_df