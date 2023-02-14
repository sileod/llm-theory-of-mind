import pandas as pd
import subprocess
import random
import csv

def execute(cmd):
    """
    Execute a bash command and return the output
    """
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, executable='/bin/zsh')
    return p.stdout.read().decode('utf-8')

def solve_problem(smcdel_problem):
    """
    Solve a problem and return the label
    """
    cmd = f'echo "{smcdel_problem}" | smcdel - 2> /dev/null | egrep "True|False"'
    output = execute(cmd)

    y = -1                      # -1 means that the problem has a syntax error in it
    if 'True' in output:
        y = 1
    elif 'False' in output:
        y = 0
    return y


if __name__ == '__main__':
    # Read the problems
    df = pd.read_csv('test001.csv')

    # Remove duplicates
    df = df.drop_duplicates()

    # Solve the problems
    df['label'] = df['problem'].apply(solve_problem)

    # Get a random sample of one problem for each premise and label
    # This is to make sure that we have a balanced dataset
    # We want to get 2 assertions for each premises
    # One true and one false
    one_of_each = df.groupby(['premises', 'label']).apply(lambda x: x.sample(1, random_state=42))
    one_of_each.rename(columns={'premises': 'prem'}, inplace=True)

    # Create the final dataframe
    final_df = pd.DataFrame(columns=['problem', 'premises', 'true_assertion', 'false_assertion', 'hypotesis', 'label'])

    # For each premises, get the true and false assertion
    for premises, group in one_of_each.groupby('prem'):
        true_assertions = group[group['label'] == 1]['assertion']
        false_assertions = group[group['label'] == 0]['assertion']

        if len(true_assertions) == 0 or len(false_assertions) == 0:
            continue

        true_assertion = true_assertions.values[0]
        false_assertion = false_assertions.values[0]

        label = group['label'].values[0]
        problem = group['problem'].values[0]

        if random.choice([True, False]):
            hypotesis = true_assertion
            label = 'entailment'
        else:
            hypotesis = false_assertion
            label = 'neutral'
            

        # Create a new dataframe with the problem and the true and false assertion
        pb_df = pd.DataFrame([[problem, premises, true_assertion, false_assertion, hypotesis, label]], columns=['problem', 'premises', 'true_assertion', 'false_assertion', 'hypotesis', 'label'])

        # Append the new dataframe to the final dataframe
        final_df = pd.concat([final_df, pb_df], ignore_index=True)

    # Save the final dataframe to a csv file
    final_df.to_csv('final_dataset001.csv', index=False, quoting=csv.QUOTE_ALL)