import random
import numpy as np

class Expression:
    """
    Represents a logical expression
    """
    def __init__(self, expr, format='smcdel'):
        self.expr = expr
        self.format = format

    def __str__(self):
        if self.format == 'smcdel':
            return f'{self.expr.to_smcdel()}'
        return f'{self.expr}'

class Operator:
    """
    Represents a logical operator
    """
    def __init__(self, symbol):
        """
        :param symbol: The symbol of the operator
        """
        self.symbol = symbol

class UnaryOperator(Operator):
    """
    Represents a unary logical operator
    """
    def __init__(self, symbol, expr):
        super().__init__(symbol)
        self.expr = expr

    def __str__(self):
        return f'{self.symbol} {self.expr}'

    def to_smcdel(self):
        return f'{self.smcdel_symbol} {self.expr.to_smcdel()}'

class Not(UnaryOperator):
    """
    Represents a logical negation
    """
    def __init__(self, expr):
        super().__init__('not', expr)
        self.smcdel_symbol = '~'

    def __str__(self):
        # Replaces is by is not to negate the expression
        return str(self.expr).replace(' is ', ' is not ')

class BinaryOperator(Operator):
    """
    Represents a binary logical operator
    """
    def __init__(self, symbol, l_expr, r_expr):
        """
        :param symbol: The symbol of the operator
        :param l_expr: The left expression
        :param r_expr: The right expression
        """
        super().__init__(symbol)
        self.l = l_expr
        self.r = r_expr

    def __str__(self):
        return f'{self.l} {self.symbol} {self.r}'

    def to_smcdel(self):
        return f'{self.l.to_smcdel()} {self.smcdel_symbol} {self.r.to_smcdel()}'

class Or(BinaryOperator):
    """
    Represents a logical disjunction
    """
    def __init__(self, l_expr, r_expr):
        """
        :param l_expr: The left expression
        :param r_expr: The right expression
        """
        super().__init__('or', l_expr, r_expr)
        self.smcdel_symbol = '|'

class And(BinaryOperator):
    """
    Represents a logical conjunction
    """
    def __init__(self, l_expr, r_expr):
        """
        :param l_expr: The left expression
        :param r_expr: The right expression
        """
        super().__init__('and', l_expr, r_expr)
        self.smcdel_symbol = '&'

class Var:
    """
    Represents a variable
    """
    def __init__(self, name, id):
        """
        :param name: The name of the variable. We use it to indicate what the variable represents. (e.g. 'Alice is muddy')
        :param id: The id of the variable. This is used to represent the variable as a number in the smcdel format.
        """
        self.name = name
        self.id = id

    def __str__(self):
        # We don't want to print the name of top and bottom
        if self.name in ['Top', 'Bottom']:
            return ''
        return f'{self.name}'

    def to_smcdel(self):
        return f'{self.id}'

class Knowledge:
    """
    Represents a knowledge expression
    """
    def __init__(self, agent, expr):
        """
        :param agent: The agent that has the knowledge
        :param expr: The expression to which the agent has the knowledge
        """
        self.agent = agent
        self.expr = expr
        self.symbol = ''
    
    def to_smcdel(self):
        return f'{self.agent} {self.symbol} {self.expr.to_smcdel()}'

    def __str__(self):
        return f'{self.agent} {self.symbol} {self.expr}'

class KnowsThat(Knowledge):
    """
    Represents a knowledge expression of the form "agent knows that"
    This implies that the agent knows that the expression is true
    """
    def __init__(self, agent, expr):
        """
        :param agent: The agent that has the knowledge
        :param expr: The expression to which the agent has the knowledge
        """
        super().__init__(agent, expr)
        self.symbol = 'knows that'

class KnowsWhether(Knowledge):
    """
    Represents a knowledge expression of the form "agent knows whether"
    This implies that the agent knows whether the expression is true or false
    """
    def __init__(self, agent, expr):
        """
        :param agent: The agent that has the knowledge
        :param expr: The expression to which the agent has the knowledge
        """
        super().__init__(agent, expr)
        self.symbol = 'knows whether'

class Announcement:
    """
    Represents an announcement
    A public announcement is of the form "It is publicly announced that"
    It tells all agents that the expression is true
    """
    def __init__(self, expr):
        """
        :param expr: The expression that is publicly announced
        """
        self.expr = expr

    def __str__(self):
        return f'It is publicly announced that {self.expr}'

    def to_smcdel(self):
        return f'[ ! {self.expr.to_smcdel()} ]'

class Law:
    """
    Represents a law
    It is used to define the initial state of the world
    It can be Top or Bottom
    """
    def __init__(self, expr):
        """
        :param expr: The expression of the law
        """
        # If the expression is Top or Bottom, we create a variable with the same name
        if expr == 'Top' or expr == 'Bottom':
            self.expr = Var(expr, expr)
        else:
            self.expr = expr

    def __str__(self):
        return f'{self.expr}.' if len(str(self.expr)) > 0 else ''

    def to_smcdel(self):
        return f'LAW {self.expr.to_smcdel()}'

class Problem:
    """
    Represents a problem
    """
    def __init__(self, **setup):

        # The format to display the problem in
        self.format = 'smcdel'

        # The variables in the problem
        self.variables = setup['variables']

        # The agents in the problem
        self.agents = setup['agents']

        # The number of announcements in the problem
        self.n_announcements = setup['n_announcements']

        # The base observation of the problem
        self.base_observation = setup['observation']

        # The law of the problem
        self.law = setup['law']
        if self.law is None:
            # If no law is given, we create a random law
            self.law = Expression(Law(random_expression(self.variables, 1)))
        
        # The observations of the problem
        self.observations = setup['matrix']
        if self.observations is None:
            self.observations = np.random.randint(2, size=(len(self.agents), len(self.variables)))

        # The announcements of the problem
        self.announcements = setup['announcements']
        if self.announcements is None:
            # If no announcements are given, we create random announcements
            self.announcements = [Expression(Announcement(random.choice([random_expression(self.variables, 1), random_knowledge(self.agents, self.variables, 0)]))) for i in range(self.n_announcements)]

        # The hypothesis of the problem
        self.hypothesis = setup['hypothesis']
        if self.hypothesis is None:
            # If no hypothesis is given, we create a random hypothesis
            self.hypothesis = Expression(random.choice([KnowsThat, KnowsWhether])(random.choice(self.agents), random_expression(self.variables, 0)))


    def get_vars(self):
        """
        :return: The variables in the problem
        """
        return self.variables

    def observations_to_str(self):
        """
        :return: The observations of the problem as a string
        """
        # We get the indices of the non-zero elements in the matrix
        # And we group them by agent
        mx = np.transpose(self.observations.nonzero())
        groupby = np.split(mx[:, 1], np.unique(mx[:,0], return_index=True)[1][1:])

        # We create the string
        result = ''
        if self.format == 'smcdel':
            result += 'OBS '
        for agent in range(len(groupby)):
            if self.format == 'smcdel':
                result += f'{self.agents[agent]}:' + ','.join([self.variables[var].to_smcdel() for var in groupby[agent]]) + ' '
            else:
                result += f'{self.agents[agent]} knows whether ' + ', whether '.join([str(self.variables[var]) for var in groupby[agent]]) + '. '
        return result

    def announcements_to_str(self):
        """
        :return: The announcements of the problem as a string
        """
        if self.format == 'smcdel':
            return ' '.join([announcement.expr.to_smcdel() for announcement in self.announcements])
        return '. '.join([str(announcement) for announcement in self.announcements]) + '.'

    def change_format(self, format):
        """
        Sets the format of the problem
        :param format: The format to change to
        """
        self.format = format
        self.law.format = format
        self.hypothesis.format = format
        for announcement in self.announcements:
            announcement.format = format

    def __str__(self):
        if self.format == 'smcdel':
            result = 'VARS ' + ','.join([var.to_smcdel() for var in self.variables]) + ' ' + self.law.expr.to_smcdel() + ' '
        elif self.format == 'natural':
            result += f'{self.law} '
        result += f'{self.observations_to_str()}'.strip()

        if self.format == 'smcdel':
            result += f' VALID? {self.announcements_to_str()} {self.hypothesis.expr.to_smcdel()}'
        elif self.format == 'natural':
            result += f'{self.announcements_to_str()} {self.hypothesis}'
        return result

def show_pb(p):
    """
    Returns a string representation of the problem
    :param p: The problem
    :return: The string representation of the problem
    """
    print(p)
    print()
    print('variables : ', p.get_vars())
    print('agents :', p.agents)
    print('law :', p.law)
    print('observations :', p.observations_to_str())
    print('announcements :', p.announcements_to_str())
    print('hypothesis :', p.hypothesis)

def random_expression(vars, depth):
    """
    Generates a random logical expression of depth depth.
    """

    if depth == 0:
        # 50% chance of negation
        if random.random() < 0.5:
            return Not(random.choice(vars))
        return random.choice(vars)

    # return random.choice([Or, And])(random_expression(vars, depth - 1), random_expression(vars, depth - 1))
    return And(random_expression(vars, depth - 1), random_expression(vars, depth - 1))
    
def random_knowledge(agents, vars, depth, exclude_agent=None):
    """
    Generates a random knowledge expression of depth depth.
    """
    knowledge_type = random.choice([KnowsThat, KnowsWhether])
    agent = random.choice(agents)

    while agent == exclude_agent:
        agent = random.choice(agents)

    if depth == 0:
        return knowledge_type(agent, random.choice(vars))

    return knowledge_type(agent, random_knowledge(agents, vars, depth - 1, agent))