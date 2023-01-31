import subprocess
import pandas as pd
import sys

# Execute a bash command and return the output
def execute(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, executable='/bin/zsh')
    return p.stdout.read().decode('utf-8')

def solve_problems_from_csv(path, result_path):
    
    print('Reading the problems from', path)

    # Read the problems
    df = pd.read_csv(path)

    print('Problems read')
    print('Processing', len(df), 'problems...')

    # Create a new dataframe to store the results
    y_df = pd.DataFrame(columns=['valid'])

    nb_processed = 0

    # For each problem, check the result
    for pb in df['problem']:

        # Prepare the command
        cmd = f'echo "{pb}" | smcdel - 2> /dev/null | egrep "True|False"'

        # Execute the command
        output = execute(cmd)

        # Determine the label
        y = -1                      # -1 means that the problem has a syntax error in it
        if 'True' in output:
            y = 1
        elif 'False' in output:
            y = 0

        # Append the label to the dataframe
        y_df = pd.concat([y_df, pd.DataFrame([y], columns=['valid'])], ignore_index=True)

        # Print the progress
        nb_processed += 1
        if nb_processed % (len(df)//10) == 0:
            print((nb_processed / len(df)) * 100, '% processed')
    
    print('Processing done')

    # Concatenate the results to the original dataframe
    df = pd.concat([df, y_df], axis=1)

    print('Saving the results to', result_path)
    # Save the dataframe to a csv file
    df.to_csv(result_path, index=False)
    print('Done')


if __name__ == '__main__':
    # Get the path from the command line
    path = sys.argv[1]
    result_path = sys.argv[2]

    # Solve the problems
    solve_problems_from_csv(path, result_path)