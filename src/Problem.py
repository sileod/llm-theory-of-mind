import random
import numpy as np

class Expression:
    def __init__(self, expr, format='smcdel'):
        self.expr = expr
        self.format = format

    def __str__(self):
        if self.format == 'smcdel':
            return f'{self.expr.to_smcdel()}'
        return f'{self.expr}'

    def get_vars(self):
        return list(set(self.expr.get_vars()))

class Operator:
    def __init__(self, symbol):
        self.symbol = symbol

    def get_vars(self):
        return self.get_vars()

class UnaryOperator(Operator):
    def __init__(self, symbol, expr):
        super().__init__(symbol)
        self.expr = expr

    def get_vars(self):
        return self.expr.get_vars()

    def __str__(self):
        return f'{self.symbol} {self.expr}'

    def to_smcdel(self):
        return f'{self.smcdel_symbol} {self.expr.to_smcdel()}'

class Not(UnaryOperator):
    def __init__(self, expr):
        super().__init__('not', expr)
        self.smcdel_symbol = '~'

class BinaryOperator(Operator):
    def __init__(self, symbol, l_expr, r_expr):
        super().__init__(symbol)
        self.l = l_expr
        self.r = r_expr

    def get_vars(self):
        return self.l.get_vars() + self.r.get_vars()

    def __str__(self):
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

class Var:
    def __init__(self, name, id):
        self.name = name
        self.id = id
        # self.to_smcdel = self.__str__

    def __str__(self):
        if self.name in ['Top', 'Bottom']:
            return ''
        return f'{self.name}'

    def to_smcdel(self):
        return f'{self.id}'

    def get_vars(self):
        if self.name in ['Top', 'Bottom']:
            return []
        return [self.name]

class Knowledge:
    def __init__(self, agent, expr):
        self.agent = agent
        self.expr = expr
        self.symbol = ''
    
    def to_smcdel(self):
        return f'{self.agent} {self.symbol} {self.expr.to_smcdel()}'

    def __str__(self):
        return f'{self.agent} {self.symbol} {self.expr}'

    def get_vars(self):
        return self.expr.get_vars()

class KnowsThat(Knowledge):
    def __init__(self, agent, expr):
        super().__init__(agent, expr)
        self.symbol = 'knows that'

class KnowsWhether(Knowledge):
    def __init__(self, agent, expr):
        super().__init__(agent, expr)
        self.symbol = 'knows whether'

class Announcement:
    def __init__(self, expr):
        self.expr = expr

    def __str__(self):
        return f'It is publicly announced that {self.expr}'

    def to_smcdel(self):
        return f'[ ! {self.expr.to_smcdel()} ]'

    def get_vars(self):
        return self.expr.get_vars()

class Law:
    def __init__(self, expr):
        if expr == 'Top' or expr == 'Bottom':
            self.expr = Var(expr, expr)
        else:
            self.expr = expr

    def __str__(self):
        return f'{self.expr}.' if len(str(self.expr)) > 0 else ''

    def to_smcdel(self):
        return f'LAW {self.expr.to_smcdel()}'

    def get_vars(self):
        return self.expr.get_vars()

class Problem:
    def __init__(self, **setup):
        self.format = 'smcdel'

        self.variables = setup['variables']
        self.agents = setup['agents']

        self.n_announcements = setup['n_announcements']

        self.base_observation = setup['observation']

        self.law = setup['law']
        if self.law is None:
            self.law = Expression(Law(random_expression(self.variables, 1)))
        
        self.observations = setup['matrix']
        if self.observations is None:
            self.observations = {agent: [random.choice(self.variables)] for agent in self.agents}

        self.announcements = setup['announcements']
        if self.announcements is None:
            self.announcements = [Expression(Announcement(random.choice([random_expression(self.variables, 1), Knowledge(random.choice(self.agents), random_expression(self.variables, 0))]))) for i in range(1)]

        self.hypothesis = setup['hypothesis']
        if self.hypothesis is None:
            self.hypothesis = random.choice([Expression(random_expression(self.variables, 1)), Expression(Knowledge(random.choice(self.agents), random_expression(self.variables, 0)))])


    def get_vars(self):
        return self.variables

    # def observations_to_str(self):
    #     result = ''
    #     if self.format == 'smcdel':
    #         result += 'OBS '

    #     for agent, exprs in self.observations.items():
    #         if self.format == 'smcdel':
    #             result += f'{agent}: ' + ','.join([expr.to_smcdel() for expr in exprs]) + ' '
    #         elif self.format == 'natural':
    #             result += f'{agent} knows whether ' + ','.join([str(expr) for expr in exprs]) + '. '
    #     return result

    def observations_to_str(self):
        mx = np.transpose(self.observations.nonzero())
        groupby = np.split(mx[:, 1], np.unique(mx[:,0], return_index=True)[1][1:])
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
        if self.format == 'smcdel':
            return ' '.join([announcement.expr.to_smcdel() for announcement in self.announcements])
        return '. '.join([str(announcement) for announcement in self.announcements]) + '.'

    def change_format(self, format):
        self.format = format
        self.law.format = format
        self.hypothesis.format = format
        # for exprs in self.observations.values():
        #     for expr in exprs:
        #         expr.format = format
        for announcement in self.announcements:
            announcement.format = format

    def __str__(self):
        result = ''
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
        return random.choice(vars)

    # 50% chance of negation
    # if random.random() < 0.5:
    #     return Not(random_expression(vars, depth - 1))

    return random.choice([Or, And])(random_expression(vars, depth - 1), random_expression(vars, depth - 1))

    