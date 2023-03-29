import random
import numpy as np
randint = np.random.randint
import inflect

number_to_words = inflect.engine().number_to_words

def replace_last_occurrence(s, x, replacement):
    return s[::-1].replace(x[::-1], replacement[::-1], 1)[::-1]

class Expression:
    def __init__(self, expr, format='smcdel'):
        self.expr = expr
        self.format = format

    def __str__(self):
        if self.format == 'smcdel':
            return f'{self.expr.to_smcdel()}'
        return f'{self.expr}'

class Operator:
    def __init__(self, symbol):
        self.symbol = symbol

class UnaryOperator(Operator):
    def __init__(self, symbol, expr):
        super().__init__(symbol)
        self.expr = expr

    def __str__(self):
        return f'{self.symbol} {self.expr}'

    def to_smcdel(self):
        return f'{self.smcdel_symbol} {self.expr.to_smcdel()}'

class Not(UnaryOperator):
    def __init__(self, expr):
        super().__init__('not', expr)
        self.smcdel_symbol = '~'

    def __str__(self):
        # Replaces is by is not to negate the expression
        return str(self.expr).replace(' is ', ' is not ')

class BinaryOperator(Operator):
    def __init__(self, symbol, l_expr, r_expr):
        super().__init__(symbol)
        self.l = l_expr
        self.r = r_expr

    def __str__(self):
        if str(self.l)==str(self.r):
            return str(self.l) # Duh
        return f'{self.l} {self.symbol} {self.r}'

    def to_smcdel(self):
        return f'{self.l.to_smcdel()} {self.smcdel_symbol} {self.r.to_smcdel()}'

class Or(BinaryOperator):
    def __init__(self, l_expr, r_expr):
        super().__init__('or', l_expr, r_expr)
        self.smcdel_symbol = '|'

class And(BinaryOperator):
    def __init__(self, l_expr, r_expr):
        super().__init__('and', l_expr, r_expr)
        self.smcdel_symbol = '&'


class Someone(Operator):
    def __init__(self, variables):
        self.variables=variables

    def to_smcdel(self):
        y = "|".join([x.to_smcdel() for x in self.variables])
        return f'({y})'

    def __str__(self):
        return str(self.variables[0]).replace("Agenta","someone")

class Everyone(Someone):
    def to_smcdel(self):
        y = "&".join([x.to_smcdel() for x in self.variables])
        return f'({y})'
    def __str__(self):
        return str(self.variables[0]).replace("Agenta","everyone")

class NotEveryone(Someone):
    def to_smcdel(self):
        y="&".join([x.to_smcdel() for x in self.variables])
        return f"(~({y}))"
    def __str__(self):
        return str(self.variables[0]).replace("Agenta","not everyone")

class Var:
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
    def __init__(self, agent, expr):
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
    def __init__(self, **setup):
        self.format = 'smcdel'
        for k,v in setup.items():
            setattr(self,k,v)

        if self.law is None:
            self.law = Expression(Law(self.random_expression(self.law_depth)))
        
        # The observations of the problem
        self.observations = setup['matrix']
        if self.observations is None:
            self.observations = randint(2, size=(len(self.agents), len(self.variables)))


        self.quantifiers = [self.someone(),self.everyone(),self.not_everyone()]

        # The announcements of the problem
        self.announcements = setup['announcements']
        if self.announcements is None:
            self.announcements = [Expression(Announcement(random.choice([
                self.random_expression(self.announcement_depth+1), 
                self.random_knowledge(self.announcement_depth)]))) for i in range(self.n_announcements)]

        # The hypothesis of the problem
        self.hypothesis = setup['hypothesis']
        if self.hypothesis is None:
            # If no hypothesis is given, we create a random hypothesis
            self.hypothesis = Expression(
                random.choice([KnowsThat, KnowsWhether])(
                    random.choice(self.agents),
                    self.random_knowledge(self.hypothesis_depth)
                ))

    def get_vars(self):
        return self.variables

    def observations_to_str(self):
        # We get the indices of the non-zero elements in the matrix
        # And we group them by agent
        mx = np.transpose(self.observations.nonzero())
        groupby = np.split(mx[:, 1], np.unique(mx[:,0], return_index=True)[1][1:])
        prefix = f"There are {number_to_words(len(self.agents))} persons. "
        # We create the string
        result = ''
        if self.format == 'smcdel':
            result += 'OBS '
        elif self.base_observation != None:
            return prefix + self.base_observation

        if self.format!='smcdel':
            result=f'{prefix} {result}'
        for agent in range(len(groupby)):
            if self.format == 'smcdel':
                result += f'{self.agents[agent]}:' + ','.join([self.variables[var].to_smcdel() for var in groupby[agent]]) + ' '
            else:
                result += f'{self.agents[agent]} knows whether ' + ', whether '.join([str(self.variables[var]) for var in groupby[agent]]) + '. '
                result = replace_last_occurrence(result,', whether',' and whether')
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

    def show_pb(self):
        print(p)
        print('variables : ', p.get_vars())
        print('agents :', p.agents)
        print('law :', p.law)
        print('observations :', p.observations_to_str())
        print('announcements :', p.announcements_to_str())
        print('hypothesis :', p.hypothesis)

    def someone(self): return Someone(self.variables)
    def everyone(self): return Everyone(self.variables)
    def not_everyone(self): return NotEveryone(self.variables)

    def random_expression(self, depth):
        if depth == 0:
            # 50% chance of negation
            random_var = random.choice(self.variables+self.quantifiers)
            if random.random() < 0.5:
                return Not(random_var)
            return random_var

        # return random.choice([Or, And])(random_expression(vars, depth - 1), random_expression(vars, depth - 1))
        return And(self.random_expression(depth - 1), self.random_expression(depth - 1))
    
    def random_knowledge(self, depth, exclude_agent=None):
        knowledge_type = random.choice([KnowsThat, KnowsWhether])
        agent = random.choice(self.agents)

        while agent == exclude_agent:
            agent = random.choice(self.agents)

        if depth == 0:
            return knowledge_type(agent, random.choice(self.variables+self.quantifiers))

        return knowledge_type(agent, self.random_knowledge(depth - 1, agent))
