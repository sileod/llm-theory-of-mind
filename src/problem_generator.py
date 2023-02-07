from generic import *
import pandas as pd
import csv

if __name__ == '__main__':
    vars = [Var(str(i)) for i in range(2)]
    agents = ['alice', 'bob']

    df = pd.DataFrame(columns=['problem', 'assertion'])
    for _ in range(20):
        vars = [Var(str(i)) for i in range(2)]

        pb = generate_problem(vars, agents, law=Expression(Law('Top')))

        smcdel_pb = str(pb)

        pb.change_format('natural')
        vars[0].name = 'alice is muddy'
        vars[1].name = 'bob is muddy'

        premises  = str(pb.law) + pb.observations_to_str() + pb.announcements_to_str()
        assertion = str(pb.assertion)

        df_pb = pd.DataFrame([[smcdel_pb, premises, assertion]], columns=['problem', 'premises', 'assertion'])


        df = pd.concat([df, df_pb], ignore_index=True)

    df.to_csv('test.csv', index=False, quoting=csv.QUOTE_ALL)