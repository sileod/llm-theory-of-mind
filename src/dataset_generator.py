from Problem import *
import pandas as pd
import csv
import numpy as np
from name_generation import n_random_first_names
from string import Template
from solver import solve
from pandarallel import pandarallel
from util import to_dropbox, get_url_of_file

if __name__ == '__main__':
    pandarallel.initialize()

    # Create the dataframe
    df = pd.DataFrame(columns=['problem', 'premise', 'hypothesis'])
    for _ in range(100):

        n_agents = 3

        # Define the agents
        agents = n_random_first_names(n_agents)

        variables = [Var(Template('$agent is muddy').substitute(agent=agents[i]), i) for i in range(len(agents))]

        # Define the setup
        forehead = {
            'variables': variables,
            'agents': agents,
            'law': Expression(Law('Top')),
            'matrix': np.ones((n_agents, n_agents)) - np.eye(n_agents),
            'announcements': [Expression(Announcement(random.choice([random_expression(variables, 1), Knowledge(random.choice(agents), random_expression(variables, 0))]))) for i in range(1)],
            # 'announcements': None,
            'hypothesis': None,
            'type': 'visual',
            'observation': '',
        }

        # Create 5 problems
        for _ in range(5):

            # Creating the problem
            pb = Problem(**forehead)

            # Get the problem in smcdel format
            smcdel_pb = str(pb)

            # Switch to natural language
            pb.change_format('natural')

            # Get the problem in natural language
            premise  = str(pb.law) + pb.observations_to_str() + pb.announcements_to_str()

            # Get the hypothesis
            hypothesis = str(pb.hypothesis)

            # Create a dataframe with the problem, the premise and the hypothesis
            df_pb = pd.DataFrame([[smcdel_pb, premise, hypothesis]], columns=['problem', 'premise', 'hypothesis'])

            # Add the problem to the main dataframe
            df = pd.concat([df, df_pb], ignore_index=True)

    # Remove duplicates
    df = df.drop_duplicates()

    # df['label'] = df['problem'].apply(solve)
    df['label'] = df['problem'].parallel_apply(solve)

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
            label = 'neutral'

        pb_df = pd.DataFrame([[problem, premises, true_hyp, false_hyp, hypothesis, label]], columns=['problem', 'premise', 'true_hypothesis', 'false_hypothesis', 'hypothesis', 'label'])
        
        final_df = pd.concat([final_df, pb_df], ignore_index=True)

    # Save the dataframe to a jsonl file
    final_df.to_json('test_refactored.jsonl', orient='records', lines=True)

    to_dropbox(final_df, '/dataset.jsonl')

    url = get_url_of_file('/dataset.jsonl')

    print(url)
