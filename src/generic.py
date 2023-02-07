import random

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
    def __init__(self, name):
        self.name = name
        # self.to_smcdel = self.__str__

    def __str__(self):
        if self.name in ['Top', 'Bottom']:
            return ''
        return f'{self.name}'

    def to_smcdel(self):
        return f'{self.name}'

    def get_vars(self):
        if self.name in ['Top', 'Bottom']:
            return []
        return [self.name]

class Knowledge:
    def __init__(self, agent, expr):
        self.agent = agent
        self.expr = expr
        self.to_smcdel = self.__str__

    def __str__(self):
        return f'{self.agent} knows whether {self.expr}'

    def get_vars(self):
        return self.expr.get_vars()

class Announcement:
    def __init__(self, expr):
        self.expr = expr

    def __str__(self):
        return f'everyone sees that {self.expr}'

    def to_smcdel(self):
        return f'[ ! {self.expr.to_smcdel()} ]'

    def get_vars(self):
        return self.expr.get_vars()

class Law:
    def __init__(self, expr):
        if expr == 'Top' or expr == 'Bottom':
            self.expr = Var(expr)
        else:
            self.expr = expr

    def __str__(self):
        return f'{self.expr}.' if len(str(self.expr)) > 0 else ''

    def to_smcdel(self):
        return f'LAW {self.expr.to_smcdel()}'

    def get_vars(self):
        return self.expr.get_vars()

class Problem:
    def __init__(self, law, observations, announcements, assertion, format='smcdel'):
        self.law = law
        self.observations = observations
        self.announcements = announcements
        self.assertion = assertion
        self.format = format
        self.agents = list(observations.keys())

    def get_vars(self):
        law_vars = self.law.get_vars()
        obs_vars = []
        for agent, exprs in self.observations.items():
            for expr in exprs:
                obs_vars += expr.get_vars()
        ann_vars = []
        for announcement in self.announcements:
            ann_vars += announcement.get_vars()
        assert_vars = self.assertion.get_vars()
        # return list(set(self.law.get_vars() + self.assertion.get_vars()))
        return list(set(law_vars + obs_vars + ann_vars + assert_vars))

    def observations_to_str(self):
        result = ''
        if self.format == 'smcdel':
            result += 'OBS '

        for agent, exprs in self.observations.items():
            if self.format == 'smcdel':
                result += f'{agent}: ' + ','.join([str(expr) for expr in exprs]) + ' '
            elif self.format == 'natural':
                result += f'{agent} knows whether ' + ','.join([str(expr) for expr in exprs]) + '. '
        return result

    def announcements_to_str(self):
        if self.format == 'smcdel':
            return ' '.join([str(announcement) for announcement in self.announcements])
        return '. '.join([str(announcement) for announcement in self.announcements]) + '.'

    def change_format(self, format):
        self.format = format
        self.law.format = format
        self.assertion.format = format
        for exprs in self.observations.values():
            for expr in exprs:
                expr.format = format
        for announcement in self.announcements:
            announcement.format = format

    def __str__(self):
        result = ''
        if self.format == 'smcdel':
            result = 'VARS ' + ','.join([str(var) for var in self.get_vars()]) + ' '
        result += f'{self.law} {self.observations_to_str()}'

        if self.format == 'smcdel':
            result += f'VALID? {self.announcements_to_str()} {self.assertion}'
        elif self.format == 'natural':
            result += f'{self.announcements_to_str()} {self.assertion}'
        return result

def show_pb(p):
    print(p)
    print()
    print('variables : ', p.get_vars())
    print('agents :', p.agents)
    print('law :', p.law)
    print('observations :', p.observations_to_str())
    print('announcements :', p.announcements_to_str())
    print('assertion :', p.assertion)

def generate_problem(vars, agents, n_annoucements=1, law=None, observations=None, announcements=None, assertion=None):
    """
    Generates a random problem with n_vars variables, n_agents agents and a random law.
    """

    if law is None:
        law = Expression(Law(random_expression(vars, 1)))

    if observations is None:
        observations = {agent: [random.choice(vars)] for agent in agents}

    if announcements is None:
        announcements = [random.choice([Expression(Announcement(random_expression(vars, 1))), Expression(Announcement(Knowledge(random.choice(agents), random_expression(vars, 0))))]) for i in range(n_annoucements + 1)]
    
    if assertion is None:
        assertion = random.choice([Expression(random_expression(vars, 1)), Knowledge(random.choice(agents), random_expression(vars, 0))])

    return Problem(law, observations, announcements, assertion)

def random_expression(vars, depth):
    """
    Generates a random logical expression of depth depth.
    """

    if depth == 0:
        return random.choice(vars)

    return random.choice([Or, And])(random_expression(vars, depth - 1), random_expression(vars, depth - 1))

if __name__ == '__main__':
    vars=[Var('1'), Var('2')]

    # p = Problem(
    #     law=Expression(
    #             Law(
    #                 Or(
    #                     vars[0],
    #                     vars[1]
    #                 )
    #             ),
    #         format),
    #     observations={
    #         'alice': [vars[0]],
    #         'bob': [vars[1]]
    #     },
    #     announcements=[
    #         Expression(
    #             Announcement(
    #                 Knowledge('alice',
    #                     vars[0]
    #                 )
    #             )),
    #         Expression(
    #             Announcement(
    #                 Or(
    #                     vars[0],
    #                     vars[1]
    #                 )
    #             ))
    #     ],
    #     assertion=Expression(
    #         Knowledge('alice',
    #             vars[1]
    #             )
    #         )
    # )

    # print('PRINTING PROBLEM IN SMCDEL\n------------------------------------')
    # show_pb(p)
    # print('------------------------------------\n')

    # p.change_format('natural')

    # print('\nPRINTING PROBLEM IN NATURAL LANGUAGE\n------------------------------------')
    # vars[0].name = 'bob is muddy'
    # vars[1].name = 'alice is muddy'
    # show_pb(p)
    # print('------------------------------------\n')


    # re = Expression(random_expression(vars, 2))

    # print(re)

    vars = [Var(str(i)) for i in range(2)]
    agents = ['alice', 'bob']

    # for _ in range(1):
    #     # rp = generate_problem(vars, agents, 1, law=Expression(Law('Top')))
    #     # rp = generate_problem(vars, agents, 1)
    #     # rp = generate_problem(vars, agents, 1, law=Expression(Law(And(vars[0], vars[1]))))
    #     rp = generate_problem(vars, agents, observations={agents[0]: [vars[0], vars[1]], agents[1]: [vars[1]]}, law=Expression(Law('Top')))
    #     print(rp)


    #     rp.change_format('natural')

    #     print(rp)

    # with open('test.txt', 'w') as f:
    #     for _ in range(100):
    #         rp = generate_problem(vars, agents, 1, law=Expression(Law('Top')))
    #         # rp.change_format('natural')
    #         f.write(f'{rp}\n')

    rp = generate_problem(vars, agents, 1, law=Expression(Law('Top')))
    rp.change_format('natural')
    vars[0].name = 'alice is muddy'
    vars[1].name = 'bob is muddy'
    print(rp)
    