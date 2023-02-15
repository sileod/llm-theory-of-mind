from Problem import *
import pandas as pd
import csv
import numpy as np

if __name__ == '__main__':
    # Define the agents
    agents = ['alice', 'bob', 'carol']

    # Create the dataframe
    df = pd.DataFrame(columns=['problem', 'premise', 'hypotesis'])
    for _ in range(100000):
        # Define the variables
        vars = [Var(str(i)) for i in range(3)]

        obs = {agent: np.random.choice(vars, 1, False) for agent in agents}

        # Generate the problem
        pb = generate_problem(vars, agents, law=Expression(Law('Top')), observations=obs)

        # Get the problem in smcdel format
        smcdel_pb = str(pb)

        # Switch to natural language
        pb.change_format('natural')

        # Define the meaning of the variables
        vars[0].name = 'alice is in the'
        vars[1].name = 'the pen is on the shelf'
        vars[2].name = 'the cat is on the table'

        situation = 'alice, bob and carol are in a room with a mirror.'

        # Get the problem in natural language
        premise  = situation + str(pb.law) + pb.observations_to_str() + pb.announcements_to_str()
        # Get the hypotesis
        hypotesis = str(pb.hypotesis)

        # Create a dataframe with the problem, the premise and the hypotesis
        df_pb = pd.DataFrame([[smcdel_pb, premise, hypotesis]], columns=['problem', 'premise', 'hypotesis'])

        # Add the problem to the main dataframe
        df = pd.concat([df, df_pb], ignore_index=True)

    # Remove duplicates
    df.drop_duplicates()

    # Save the dataframe to a csv file
    # df.to_csv('test002.csv', index=False, quoting=csv.QUOTE_ALL)
    df.to_json('test002.jsonl', orient='records', lines=True)
