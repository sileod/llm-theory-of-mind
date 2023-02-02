import pandas as pd
import numpy as np
import re
from util import to_dropbox

with open('objects.txt', 'r') as f:
    objects = f.read().split('\n')

with open('token.txt', 'r') as f:
    token = f.read()

vars_regex  = re.compile(r'VARS \d(,\d)*')
law_regex   = re.compile(r'LAW [a-zA-Z0-9]+')
obs_regex   = re.compile(r'OBS ([a-zA-Z]+:\d(,\d)* )+')
valid_regex = re.compile(r'VALID\? \[ ! .+ \] (~ )?\( .+ \)')

def translate_vars(vars_string):
    """
    :param vars_string: string of the form 'VARS 1,2,3'
    :return: natural language translation of the variables declaration
    """

    # Getting the different variables
    vars = vars_string.replace('VARS ', '').split(',')

    # Randomly choosing objects for each variable
    objs = np.random.choice(objects, len(vars), replace=False)

    # Creating a dictionary to map variables to objects
    objects_dict = dict(zip(vars, objs))

    # Creating the natural language translation
    result = 'There is '
    for i in range(len(objs)):
        result += f'a {objs[i]}'
        if i < len(objs) - 2:
            result += ', '
        if i == len(objs) - 2:
            result += ' and '
    result += ' in the room. Each of them is either white or black.'

    return result.strip(), objects_dict
    
def translate_obs(obs_string, objects_dict):
    """
    :param obs_string: string of the form 'OBS A:1,2,3 B:4,5,6'
    :param objects_dict: dictionary mapping variables to objects
    :return: natural language translation of the observations
    """

    # Getting the different observations
    obs_string = obs_string.strip()
    obs_string = obs_string.replace('OBS ', '')
    obs = obs_string.split(' ')

    # Creating the natural language translation
    result = ''
    for o in obs:
        # Getting the name of the observer and the objects he sees
        name, values = o.split(':')
        # Splitting the objects
        values = values.split(',')

        result += f'{name} sees the '
        for i in range(len(values)):
            result += f'{objects_dict[values[i]]}'
            if i < len(values) - 2:
                result += ', '
            if i == len(values) - 2:
                result += ' and the '
        result += '. '

    return result.strip()

def declaration_to_str(declaration, objects_dict, negation, question=False):
    """
    :param declaration: string of the form 'A knows whether 1'
    :param objects_dict: dictionary mapping variables to objects
    :param negation: whether the declaration is negated or not
    :param question: whether the declaration is a question or not
    :return: natural language translation of the declaration
    """
    # Getting the agent and the variable
    agent, var = declaration.split(' knows whether ')

    # Creating the natural language translation
    result = f'{agent} knows what color the {objects_dict[var]} is. '

    # Negating the statement if necessary
    if negation:
        result = result.replace('knows', 'does not know')

    # Turning the statement into a question if necessary
    if question:
        result = f'Is it true that {result.replace(".", "?")}'

    return result
    
def formula_to_str(formula, objects_dict, negation):
    """
    :param formula: string of the form ' 1 | 2 & 3 '
    :param objects_dict: dictionary mapping variables to objects
    :param negation: whether the formula is negated or not
    :return: natural language translation of the formula
    """
    # Replacing operators and splitting the formula
    formula.replace('|', 'or').replace('&', 'and')
    formula = formula.split(' ')
    
    # Creating the natural language translation
    result = ''
    for elem in formula:
        # If the element is a variable, we replace it with the corresponding object
        if elem in objects_dict:
            result += f'the {objects_dict[elem]} is {"white" if negation else "black"}'
        # If the element is an operator, we add it to the result
        else:
            if negation:
                if elem == 'and':
                    result += ' or '
                elif elem == 'or':
                    result += ' and '
            else:
                result += f' {elem} '

    return result[0].upper() + result[1:] + '. '

def translate_valid(valid_string, objects_dict):
    """
    :param valid_string: string of the form 'VALID? [ ! A knows whether 1 ] ( 1 | 2 & 3 )'
    :param objects_dict: dictionary mapping variables to objects
    """
    # Regexes to find the different parts of the string
    announcement_regex = re.compile(r'\[ ! [^\]]+ \]')
    statement_regex = re.compile(r'\] (~ )?\( .+ \)')

    # Getting the different announcements
    announcements = announcement_regex.findall(valid_string)

    result = ''
    # Creating the natural language translation of the announcements
    for announcement in announcements:
        negation = '~' in announcement
        announcement = announcement.replace(' ~', '').replace('[ ! ( ', '').replace(' ) ]', '')

        if ' knows whether ' in announcement:
            result += declaration_to_str(announcement, objects_dict, negation)
        else:
            result += formula_to_str(announcement, objects_dict, negation)

    # Getting the statement
    statement = statement_regex.search(valid_string).group(0)
    negation = '~' in statement
    statement = statement.replace('] ' + ('~ ' if negation else ''), '').replace('( ', '').replace(' )', '')

    if ' knows whether ' in statement:
        result += declaration_to_str(statement, objects_dict, negation, question=True)
    else:
        result += formula_to_str(statement, objects_dict, negation)

    return result.strip()

def translate_problem(problem):
    """
    :param problem: string of the form 'VARS A:1,2,3 B:4,5,6 OBS A:1,2,3 B:4,5,6 VALID? [ ! A knows whether 1 ] ( 1 | 2 & 3 )'
    :return: natural language translation of the problem
    """
    vars_str, o = translate_vars(vars_regex.search(problem).group(0))
    obs_str = translate_obs(obs_regex.search(problem).group(0), o)
    valid_str = translate_valid(valid_regex.search(problem).group(0), o)

    return ' '.join([vars_str, obs_str, valid_str])


if __name__ == '__main__':
    df = pd.read_csv('https://www.dropbox.com/s/to69yj7vi1l4jvm/dataset_solved.csv?dl=1')
    df['translation'] = df['problem'].apply(translate_problem)
    to_dropbox(df, '/translated_dataset.csv', token)