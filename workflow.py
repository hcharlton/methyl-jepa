import os
from gwf import Workflow
from gwf import AnonymousTarget

# 1. CONFIG
CONFIG = {
    'project_root': '/home/chcharlton/mutationalscanning/Workspaces/chcharlton/methyl-jepa',
    'gdk_account': 'mutationalscanning',
    'model_config_path': 'methyl_jepa/config.yaml',

    # ---Labeled Data (Training) ---
    'pos_bam_path': "data/00_raw/labeled/methylated_hifi_reads.bam",
    'neg_bam_path': "data/00_raw/labeled/unmethylated_hifi_reads.bam", 
    'train_ds_path': 'data/01_processed/train_sets/pacbio_standard_train.parquet',
    'test_ds_path': 'data/01_processed/train_sets/pacbio_standard_test.parquet',
    'norm_stats_path': 'data/02_analysis/norm_stats.yaml',
    'train_prop': 0.8,
    'n_reads': 600_000,
    'context': 32,

    # --- Unlabeled Data (Inference) ---
    # martin
    'martin_bam_path': 'data/00_raw/unlabeled/martin_kinetics_diploid.bam',
    'martin_ds_path': 'data/01_processed/inference_sets/martin.parquet',
    'martin_optional_tags': ['np', 'sm', 'sx'],
    # da1
    'da1_bam_path': 'data/00_raw/unlabeled/da1_kinetics_diploid.bam',
    'da1_ds_path': 'data/01_processed/inference_sets/da1.parquet',
    'da1_optional_tags': ['np'],
    # positive_data_target, negative_data_target
    'positive_ds_path' : 'data/01_processed/inference_sets/positive.parquet',
    'negative_ds_path' : 'data/01_processed/inference_sets/negative.parquet',
    'posneg_optional_tags': ['np', 'sm', 'sx'],
    # testset_inference_dataset_target
    'testset_inference_ds_path': 'data/01_processed/inference_sets/testset_inference_dataset.parquet',
    # --- SSL DATA ---
    # da1
    'da1_ssl_ds_path': "data/01_processed/ssl_sets/da1.parquet",
    # --- Model & Training ---
    'output_artifact_path': 'models/v_1_model_artifact.pt',
    'output_artifact_path_cpu': 'models/v1_model_artifact_cpu.pt',
    'output_log_path': 'models/v1_model_log.csv',
    'output_log_path_cpu': 'models/v1_model_log_cpu.csv',
    'num_workers': 8,

    # --- Inference ---
    'hello_world_inference_output_path': 'results/hello_world_inference.parquet',
    'martin_inference_output_path': 'results/martin_inference.parquet',
    'da1_inference_output_path': 'results/da1_inference.parquet'
    }

# resolve paths helper
def p(path):
    return os.path.join(CONFIG['project_root'], path)

# SLURM backend gwf worker
gwf = Workflow(defaults={'account': CONFIG['gdk_account']})

def create_train_test_datasets(pos_bam, neg_bam, train_out, test_out, context, train_prop, n_reads=600_000):
    """Creates train and test datasets"""
    inputs = {'pos_bam': pos_bam, 'neg_bam': neg_bam}
    outputs = {'train_ds': train_out, 'test_ds': test_out}
    options = {'cores': 32, 'memory': '256gb', 'walltime': '02:00:00'}
    spec=f"""
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate methyl-jepa
    cd {p('')}
    python -m scripts.make_labeled_dataset \\
        --pos-bam {pos_bam} \\
        --neg-bam {neg_bam} \\
        --train-output {train_out} \\
        --test-output {test_out} \\
        --n-reads {n_reads} \\
        --context {context} \\
        --train-prop {train_prop}
        """
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def create_inference_dataset(unlabeled_bam, out_file, n_reads, context, optional_tags):
    inputs = {'in_bam': unlabeled_bam}
    outputs = {'out_parquet': out_file}

    tags_str = ' '.join(optional_tags)

    options = {'cores': 32, 'memory': '512gb', 'walltime': '03:00:00'}
    spec = f"""
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate methyl-jepa
    cd {p('')}
    python -m scripts.make_inference_dataset \\
        -i {unlabeled_bam} \\
        -n {n_reads} \\
        -c {context} \\
        -o {out_file} \\
        -t {tags_str}
    """
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def create_testset_inference_dataset(pos_ds, neg_ds, test_ds, out_ds):
    inputs = {'positive_dataset': pos_ds, 
              'negative_dataset': neg_ds, 
              'test_dataset': test_ds}
    outputs = {'output_dataset': out_ds}
    options = {'cores': 16, 'memory': '256gb', 'walltime': '01:00:00'}
    spec = f"""
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate methyl-jepa
    cd {p('')}
    python -m scripts.make_testset_inference_dataset \\
        -p {pos_ds} \\
        -n {neg_ds} \\
        -t {test_ds} \\
        -o {out_ds}
    """
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)


def compute_norm_stats(train_parquet_path, output_json_path):
    """Calculates mean/std from the training data."""
    inputs = {'train_set': train_parquet_path}
    outputs = {'stats_file': output_json_path}
    options = {'cores': 16, 
               'memory': '128gb', 
               'walltime': '00:10:00'}
    spec = f"""
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate methyl-jepa
    cd {p('')}
    python -m scripts.compute_norm_stats \\
        -i {train_parquet_path} \\
        -o {output_json_path}
    """
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def train(config_path, train_data_path, test_data_path, stats_path, output_artifact_path, output_log_path, num_workers, gpu = True):
    inputs = {'config': config_path, 
              'train_data_path': train_data_path,
              'test_data_path': test_data_path,
              'stats_path': stats_path,
              }
    outputs = {'artifact': output_artifact_path}
    if gpu: 
        options = {'cores': num_workers, 
                'memory': '128gb', 
                'walltime': '08:00:00',
                'gres': 'gpu:1',
                'account': f'{CONFIG['gdk_account']} --partition=gpu'}
        spec  = f"""
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate methyl-dev
        cd {p('')}
        python -m methl_jepa.train \\
        --config-path  {config_path} \\
        --train-data-path {train_data_path} \\
        --test-data-path {test_data_path} \\
        --stats-path {stats_path} \\
        --output-artifact-path {output_artifact_path} \\
        --output-log-path {output_log_path} \\
        --num-workers {num_workers}
        """
    else: 
        # this is for testing only
        options = {'cores': num_workers, 
                'memory': '64gb', 
                'walltime': '00:30:00'
                }
        spec  = f"""
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate methyl-dev
        cd {p('')}
        python -m methyl_jepa.train \\
        --config-path  {config_path} \\
        --train-data-path {train_data_path} \\
        --test-data-path {test_data_path} \\
        --stats-path {stats_path} \\
        --output-artifact-path {output_artifact_path} \\
        --output-log-path {output_log_path} \\
        --num-workers {num_workers}
        """
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def infer(artifact_path, data_path, stats_path, output_path, num_workers, row_groups, gpu=True):
    inputs = {'artifact': artifact_path,
              'data_path': data_path,
              'stats_path': stats_path
              }
    outputs = {'inference_out': output_path}
    if gpu:
        options = {'cores': num_workers, 
                    'memory': '128gb', 
                    'walltime': '1:00:00',
                    'gres': 'gpu:1',
                    'account': f'{CONFIG['gdk_account']} --partition=gpu'}
        spec  = f"""
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate methyl-dev
        cd {p('')}
        python -m methyl_jepa.infer \\
        --artifact-path {artifact_path}\\
        --data-path {data_path} \\
        --stats-path {stats_path} \\
        --output-path {output_path} \\
        --num-workers {num_workers} \\
        --restrict-row-groups {row_groups}
        """
    else:
        options = {'cores': num_workers, 
                    'memory': '128gb', 
                    'walltime': '0:30:00'}
        spec  = f"""
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate methyl-dev
        cd {p('')}
        python -m methyl_jepa.infer \\
        --artifact-path {artifact_path}\\
        --data-path {data_path} \\
        --stats-path {stats_path} \\
        --output-path {output_path} \\
        --num-workers {num_workers} \\
        --restrict-row-groups {row_groups}
        """
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

### ---------- WORKFLOW GRAPH ------------

### DATA ####

# datasets
train_data_target = gwf.target_from_template(
    name='create_labeled_datasets',
    template=create_train_test_datasets(
        pos_bam=p(CONFIG['pos_bam_path']),
        neg_bam=p(CONFIG['neg_bam_path']),
        train_out=p(CONFIG['train_ds_path']),
        test_out=p(CONFIG['test_ds_path']),
        n_reads=CONFIG['n_reads'],
        context=CONFIG['context'],
        train_prop=CONFIG['train_prop']
    )
)

positive_data_target = gwf.target_from_template(
    name='create_positive_dataset',
    template=create_inference_dataset(
        unlabeled_bam=CONFIG['pos_bam_path'],
        out_file=CONFIG['positive_ds_path'],
        n_reads=0,
        context=CONFIG['context'],
        optional_tags=CONFIG['posneg_optional_tags']
    )
)

negative_data_target = gwf.target_from_template(
    name='create_negative_dataset',
    template=create_inference_dataset(
        unlabeled_bam=CONFIG['neg_bam_path'],
        out_file=CONFIG['negative_ds_path'],
        n_reads=0,
        context=CONFIG['context'],
        optional_tags=CONFIG['posneg_optional_tags']
    )
)

testset_inference_data_target = gwf.target_from_template(
    name='create_testset_inference_dataset',
    template=create_testset_inference_dataset(
        pos_ds=positive_data_target.outputs['out_parquet'], 
        neg_ds=negative_data_target.outputs['out_parquet'], 
        test_ds=train_data_target.outputs['test_ds'], 
        out_ds=CONFIG['testset_inference_ds_path']
        )
)

martin_data_target=gwf.target_from_template(
    name='create_martin_dataset',
    template=create_inference_dataset(
        unlabeled_bam=CONFIG['martin_bam_path'],
        out_file=CONFIG['martin_ds_path'],
        n_reads=1_000_000,
        context=CONFIG['context'],
        optional_tags=CONFIG['martin_optional_tags']
    )
)

da1_data_target = gwf.target_from_template(
    name='create_da1_dataset',
    template=create_inference_dataset(
        unlabeled_bam=CONFIG['da1_bam_path'],
        out_file=CONFIG['da1_ds_path'],
        n_reads=1_000_000,
        context=CONFIG['context'],
        optional_tags=CONFIG['da1_optional_tags']
    )
)

# norm stats
stats_target = gwf.target_from_template(
    name='compute_stats',
    template=compute_norm_stats(
        train_parquet_path=train_data_target.outputs['train_ds'],
        output_json_path=p(CONFIG['norm_stats_path'])
    )
)

### TRAINING ###
train_target_ss_v01 = gwf.target_from_template(
    name='train_ss_v01',
    template=train(
        config_path=CONFIG['config_path'],
        train_data_path=train_data_target.outputs['train_ds'],
        test_data_path=train_data_target.outputs['test_ds'],
        stats_path=stats_target.outputs['stats_file'],
        output_artifact_path=CONFIG['output_artifact_path'],
        output_log_path=CONFIG['output_log_path'],
        num_workers=CONFIG['num_workers']
    )
)

# train_target_cpu_test = gwf.target_from_template(
#     name='train_cpu',
#     template=train(
#         config_path=CONFIG['config_path'],
#         train_data_path=train_data_target.outputs['train_ds'],
#         test_data_path=train_data_target.outputs['test_ds'],
#         stats_path=stats_target.outputs['stats_file'],
#         output_artifact_path=CONFIG['output_artifact_path_cpu'],
#         output_log_path=CONFIG['output_log_path_cpu'],
#         num_workers=CONFIG['num_workers'],
#         gpu=False
#     )
# )
 
### INFERENCE ####
# hello_world_infer_target = gwf.target_from_template(
#     name = 'hello_world_infer',
#     template = infer(artifact_path=train_target_cpu_test.outputs['artifact'], 
#                      data_path=train_data_target.outputs['test_ds'], 
#                      stats_path=stats_target.outputs['stats_file'], 
#                      output_path='results/hello_world_inference.parquet', 
#                      num_workers=CONFIG['num_workers'], 
#                      row_groups=1,
#                      gpu=False)
# )

martin_inference_target = gwf.target_from_template(
    name = 'martin_inference',
    template = infer(artifact_path=train_target_ss_v01.outputs['artifact'], 
                     data_path=martin_data_target.outputs['out_parquet'], 
                     stats_path=stats_target.outputs['stats_file'], 
                     output_path=CONFIG['martin_inference_output_path'], 
                     num_workers=CONFIG['num_workers'],
                     row_groups=0
                     )
)

testset_inference_target = gwf.target_from_template(
    name = 'testset_inference',
    template = infer(artifact_path=train_target_ss_v01.outputs['artifact'], 
                     data_path=train_data_target.outputs['test_ds'], 
                     stats_path=stats_target.outputs['stats_file'], 
                     output_path='results/testset_inference.parquet', 
                     num_workers=CONFIG['num_workers'], 
                     row_groups=0,
                     gpu=True)
)

testset_inference_debug_target = gwf.target_from_template(
    name = 'testset_inference_debug',
    template = infer(artifact_path=train_target_ss_v01.outputs['artifact'], 
                     data_path=testset_inference_data_target.outputs['output_dataset'], 
                     stats_path=stats_target.outputs['stats_file'], 
                     output_path='results/testset_inference_debug.parquet', 
                     num_workers=CONFIG['num_workers'], 
                     row_groups=0,
                     gpu=True)
)

da1_inference_target = gwf.target_from_template(
    name = 'da1_inference',
    template = infer(artifact_path=train_target_ss_v01.outputs['artifact'], 
                     data_path=da1_data_target.outputs['out_parquet'], 
                     stats_path=stats_target.outputs['stats_file'], 
                     output_path=CONFIG['da1_inference_output_path'], 
                     num_workers=CONFIG['num_workers'],
                     row_groups=0
                     )
)