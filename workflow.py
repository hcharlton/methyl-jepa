import os
from gwf import Workflow
from gwf import AnonymousTarget

# 1. CONFIG
CONFIG = {
    'project_root': '/home/chcharlton/mutationalscanning/Workspaces/chcharlton/methyl-jepa',
    'gdk_account': 'mutationalscanning',
    'config_path': 'src/config.yaml',

    # --- Data Creation ---
    'pos_bam_path': "data/raw/methylated_hifi_reads.bam",
    'neg_bam_path': "data/raw/unmethylated_hifi_reads.bam", 
    'train_ds_path': 'data/processed/pacbio_standard_train.parquet',
    'test_ds_path': 'data/processed/pacbio_standard_test.parquet',
    'norm_stats_path': 'data/processed/norm_stats.yaml',
    'train_prop': 0.8,
    'n_reads': 10,
    'context_size': 32,

    # --- Model & Training ---
    'model_architecture': 'MethylCNNv1',
    'feature_set': 'hemi', 
    'epochs': 10,
    'batch_size': 8192,
    'learning_rate': 0.001,
    'artifact_path': 'models/train_model_test_output.pt',
    'num_workers': 8,

    # --- Inference ---
    'inference_bam_path': '~/mutationalscanning/DerivedData/bam/HiFi/chimp/martin/kinetics/martin_kinetics_diploid.bam',
    'n_reads_inference': 1_000_000,
    'inference_dataset_name': 'martin_ss_v1'
}

# resolve paths helper
def p(path):
    return os.path.join(CONFIG['project_root'], path)

# SLURM backend gwf worker
gwf = Workflow(defaults={'account': CONFIG['gdk_account']})

def create_train_test_datasets(pos_bam, neg_bam, train_out, test_out, n_reads, context, train_prop):
    """Creates train and test datasets"""
    inputs = {'pos_bam': pos_bam, 'neg_bam': neg_bam}
    outputs = {'train_ds': train_out, 'test_ds': test_out}
    options = {'cores': 16, 'memory': '64gb', 'walltime': '01:00:00'}
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


def compute_norm_stats(train_parquet_path, output_json_path):
    """Calculates mean/std from the training data."""
    inputs = {'train_set': train_parquet_path}
    outputs = {'stats_file': output_json_path}
    options = {'cores': 8, 
               'memory': '16gb', 
               'walltime': '00:00:50'}
    spec = f"""
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate methyl-jepa
    cd {p('')}
    python -m scripts.compute_norm_stats \\
        -i {train_parquet_path} \\
        -o {output_json_path}
    """
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def train(config_path, train_data_path, test_data_path, stats_path, output_path, num_workers):
    inputs = {'config': config_path, 
              'train_data_path': train_data_path,
              'test_data_path': test_data_path,
              'stats_path': stats_path,}
    outputs = {'artifact': output_path}
    options = {'cores': 8, 
               'memory': '16gb', 
               'walltime': '00:00:50'}
    spec  = f"""
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate methyl-dev
    cd {p('')}
    python -m src.train \\
    --config-path  {config_path} \\
    --train-data-path {train_data_path} \\
    --test-data-path {test_data_path} \\
    --stats-path {stats_path} \\
    --output-path {output_path} \\
    --num-workers {num_workers}
    """
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

# def infer

### WORKFLOW GRAPH

# train/test datasets
data_target = gwf.target_from_template(
    name='create_labeled_datasets',
    template=create_train_test_datasets(
        pos_bam=p(CONFIG['pos_bam_path']),
        neg_bam=p(CONFIG['neg_bam_path']),
        train_out=p(CONFIG['train_ds_path']),
        test_out=p(CONFIG['test_ds_path']),
        n_reads=CONFIG['n_reads'],
        context=CONFIG['context_size'],
        train_prop=CONFIG['train_prop']
    )
)

# normalization stats
stats_target = gwf.target_from_template(
    name='compute_stats',
    template=compute_norm_stats(
        train_parquet_path=data_target.outputs['train_ds'],
        output_json_path=p(CONFIG['norm_stats_path'])
    )
)

# train a model
train_target = gwf.target_from_template(
    name='train',
    template=train(
        config_path=CONFIG['config_path'],
        train_data_path=data_target.outputs['train_ds'],
        test_data_path=data_target.outputs['test_ds'],
        stats_path=stats_target.outputs['stats_file'],
        output_path=CONFIG['artifact_path'],
        num_workers=CONFIG['num_workers']
    )
)

# model training
# inference