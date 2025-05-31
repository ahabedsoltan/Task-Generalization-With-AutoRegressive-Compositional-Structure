{
    'dataset_type': 'SyntheticMultiLangRandDataset',
    'n_samples': 100000, # number of examples to generate
    'val_samples': 512,
    'n_steps': 6, #  number of languages per line (must be <= num_languages)
    'n_lines': 4, # how many lines per example (each line has tokens in different languages)
    'n_tasks': 10, # number of language combinations per split
    'num_languages': 6, # number of languages to include
    'seed': 0
}