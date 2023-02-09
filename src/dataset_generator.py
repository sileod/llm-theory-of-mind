from Problem import *
import pandas as pd
import csv

if __name__ == '__main__':
    # Define the agents
    agents = ['alice', 'bob']

    # Create the dataframe
    df = pd.DataFrame(columns=['problem', 'premises', 'assertion'])
    for _ in range(100000):
        # Define the variables
        vars = [Var(str(i)) for i in range(2)]

        # Generate the problem
        pb = generate_problem(vars, agents, law=Expression(Law('Top')))

        # Get the problem in smcdel format
        smcdel_pb = str(pb)

        # Switch to natural language
        pb.change_format('natural')

        # Define the meaning of the variables
        vars[0].name = 'alice is muddy'
        vars[1].name = 'bob is muddy'

        # Get the problem in natural language
        premises  = str(pb.law) + pb.observations_to_str() + pb.announcements_to_str()
        # Get the assertion
        assertion = str(pb.assertion)

        # Create a dataframe with the problem, the premises and the assertion
        df_pb = pd.DataFrame([[smcdel_pb, premises, assertion]], columns=['problem', 'premises', 'assertion'])

        # Add the problem to the main dataframe
        df = pd.concat([df, df_pb], ignore_index=True)

    # Remove duplicates
    df.drop_duplicates()

    # Save the dataframe to a csv file
    df.to_csv('test001.csv', index=False, quoting=csv.QUOTE_ALL)