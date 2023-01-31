from regex_based_generator import generate_problems_to_csv
from solver import solve_problems_from_csv

def generate_dataset(result_path, n_problems):
    generate_problems_to_csv(
        result_path=result_path.replace('.csv', '_unsolved.csv'),
        n_problems=n_problems,
        n_vars=3,
        n_agents=3,
        n_observations=2,
        n_announcements=1,
        statement_size=3
    )

    url_solved = solve_problems_from_csv(
        path=result_path.replace('.csv', '_unsolved.csv'),
        result_path=result_path.replace('.csv', '_solved.csv')
    )

    url_solved = url_solved.replace('dl=0', 'dl=1')

if __name__ == '__main__':
    generate_dataset('dataset.csv', 200)