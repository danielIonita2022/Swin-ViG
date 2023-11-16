import plan_and_preprocess_api as plan_and_preprocess

if __name__ == "__main__":
    dataset_id = [501]
    print('Started plan and preprocess')
    plan_and_preprocess.extract_fingerprints(dataset_id)
    print('Finished plan and preprocess')
    print('Started planning experiments')
    plan_and_preprocess.plan_experiments(dataset_id)
    print('Finished planning experiments')
    print('Started preprocessing')
    configs = ['2d']
    num_processes = [8]
    plan_and_preprocess.preprocess(dataset_id, configurations=configs, num_processes=num_processes)
