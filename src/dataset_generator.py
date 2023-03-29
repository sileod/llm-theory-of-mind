from Problem import Problem, Var, Expression, Announcement, Law
import pandit as pd
import numpy as np
import random
import re, string
from util import n_random_first_names, solve, class_to_dict
from string import Template
from pandarallel import pandarallel
from tqdm.auto import tqdm
tqdm.pandas()
from sorcery import dict_of
import copy
import jinja2
from dataclasses import dataclass
randint = np.random.randint


def name_agents(s,agent_names):
    agents=[f'Agent{i}' for i in string.ascii_lowercase[:len(agent_names)]]
    for id, name in zip(agents, agent_names):
        s=s.replace(id, name)
    assert "Agent" not in s
    return s 

def postprocess(s):
    s=s.replace('.', '. ')
    s=".".join(s.split('.')).replace('. .','.')
    s=" ".join(s.split())
    if not s.endswith('.'):
        s=s+'.'
    return s

def postprocess_hyp(s, seed=None):
    random.seed(seed)
    s=s.replace('knows', jinja2.Template(
        "{{ ['can know'] | random }}"
        ).render())
    s=s.replace('can know', jinja2.Template(
        "{{ ['can now know'] | random }}"
        ).render(),1)
    return s


@dataclass
class Setup:
    n_announcements:int=3
    n_agents:int=3
    hypothesis_depth:int=0
    announcement_depth:int=1
    variables=None
    agents=None
    announcements= None
    hypothesis= None
    type=None
    base_observation=None
    law=Expression(Law('Top'))


    def __post_init__(self):
        self.agents = [f'Agent{string.ascii_lowercase[i]}' for i in range(self.n_agents)]


clique = "Everyone is visible to others."

class forehead(Setup):
    base_observation = clique
    variables_template="$agent's forehead  is muddy"
    type='visual'
    def __post_init__(self):
        super().__post_init__()
        n_agents=self.n_agents
        self.matrix= np.ones((n_agents, n_agents)) - np.eye(n_agents)

class forehead_mirror(forehead):
    base_observation=f'{clique} There is a mirror in the room. '
    def __post_init__(self):
        super().__post_init__()
        n_agents=self.n_agents
        self.matrix= np.ones((n_agents, n_agents))

class arm(forehead_mirror):
    base_observation=clique
    variables_template="$agent's arm is muddy"

class internal(Setup):
    base_observation=clique
    variables_template='$agent is thirsty'
    def __post_init__(self):
        super().__post_init__()
        self.matrix= np.eye(self.n_agents)
    type='mental'

class explicit(Setup):
    variables_template='$agent is muddy'
    matrix=None


def generate(Npb = 1000, Nvariations = 50, scale=5,hypothesis_depth=1):
    # Initialize the parallelization
    # It is used to solve the problems
    pandarallel.initialize(verbose=0, progress_bar=True, nb_workers=8)

    # Create the dataframe
    df = pd.DataFrame(columns=['problem', 'premise', 'hypothesis'])

    problems=[]
    # Generate the problems
    for _ in tqdm(range(Npb)):
        setups = [forehead, forehead_mirror,  arm, internal]+[explicit]*3
        setups = [class_to_dict(x(
            n_agents=randint(2,scale),
            n_announcements=scale,
            hypothesis_depth=randint(0,hypothesis_depth),
        )) for x in setups]
        setup = copy.deepcopy(np.random.choice(setups))
        setup['n_announcements'] =n_announcements= randint(0,setup['n_announcements'])
        setup['variables'] = [Var(Template(setup['variables_template']).substitute(agent=setup['agents'][i]), i+1) for i in range(len(setup['agents']))]
        base_pb = Problem(**setup)
        setup['announcements'] = [Expression(Announcement(base_pb.someone()))]+[Expression(Announcement(random.choice([
            base_pb.random_expression(1), base_pb.random_knowledge(0)])))
            for _ in range(setup['n_announcements'])
        ]

        pbcheck = Problem(**setup)
        pbcheck.hypothesis = Expression(Var('0', 0))
        pbcheck = str(pbcheck).replace('VARS ', 'VARS 0,')

        for _ in range(Nvariations):
            pb = Problem(**setup) # Random hypothesis !
            smcdel_problem = str(pb)
            pb.change_format('natural')
            premise  = pb.observations_to_str() + str(pb.law) + pb.announcements_to_str()
            hypothesis = str(pb.hypothesis)
            # Create a dataframe with the problem, the premise and the hypothesis
            problems+=[dict_of(smcdel_problem,n_announcements,pbcheck,premise,hypothesis,setup=setup['name'],
             hypothesis_depth=setup['hypothesis_depth'],   
             n_agents=setup['n_agents'])]

    df = pd.DataFrame(problems)
    pb = df[['pbcheck']].drop_duplicates().set_index('pbcheck')
    pb['valid']=list(pd.Series(pb.index).parallel_apply(solve))
    valid = pb['valid'].to_dict()
    df=df[df.pbcheck.map(lambda x: not valid[x])]

    df['label'] = df['smcdel_problem'].parallel_apply(solve)
    df=df[df.label>-1]
    df=df.groupby('premise').agg(list).reset_index().\
        sieve(label=lambda x:len(set(x))>1)

    def sample(x):
        zeros=[i for i,val in enumerate(x.label) if val==0]
        ones =[i for i,val in enumerate(x.label) if val==1]

        heuristic = lambda x: random.random()+(x.count('whether')*+0.5)
        biased_sample = lambda x: sorted(x, key=heuristic)[0]

        x['true_hypothesis']=biased_sample([x.hypothesis[i] for i in ones])
        x['false_hypothesis']=biased_sample([x.hypothesis[i] for i in zeros])
        x['setup']=x['setup'][0]
        if random.choice([0, 1]):
            x['hypothesis'] = x['true_hypothesis']
            x['label'] = 'entailment'
        else:
            x['hypothesis'] = x['false_hypothesis']
            x['label'] = 'not_entailment'
        return x

    df=df.explode(['smcdel_problem','pbcheck','hypothesis','label','hypothesis_depth','setup'])
    #df=df.apply(sample,axis=1)
    df=df.drop_duplicates(subset=['premise', 'hypothesis']).reset_index(drop=True)
    # De-anonymize the names
    for k in ['n_announcements','n_agents']:
        df[k]=df[k].map(lambda x:x[0])
    df['names']= df.apply(lambda x :n_random_first_names(x.n_agents),axis=1)

    for k in ['premise','hypothesis','true_hypothesis','false_hypothesis']:
        if k not in df:
            continue
        df[k] = df.apply(lambda x: name_agents(x[k], x['names']), axis=1)
        df[k] = df[k].map(postprocess)
        if 'hyp' in k:
            df[k] = df.apply(lambda x: postprocess_hyp(x[k], seed=x['premise']),axis=1)

    df.to_json('mindgames.jsonl', orient='records', lines=True)
    return df