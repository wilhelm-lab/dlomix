import os
import tensorflow as tf
import argparse
import sys
from pyarrow.parquet import ParquetFile
sys.path.append('../../bmpc_shared_scripts/prepare_dataset')
from get_updated_alphabet import get_modification
from dlomix.interface.oktoberfest_interface import load_keras_model, process_dataset, save_keras_model
from dlomix.refinement_transfer_learning.automatic_rl_tl import AutomaticRlTlTraining, AutomaticRlTlTrainingConfig


def main():
    parser = argparse.ArgumentParser(prog='Test single small PTMs')
    parser.add_argument('--parquet', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--improve', action='store_true')

    args = parser.parse_args()

    file = ParquetFile(args.parquet)
    modifications = set()
    for batch in file.iter_batches():
        for cur_seq in batch['modified_sequence']:
            cur_mods = get_modification(str(cur_seq))
            modifications |= set(cur_mods)
    
    model = load_keras_model('baseline')
    dataset = process_dataset(
        parquet_file_path=args.parquet,
        model=model,
        modifications=list(modifications),
        ion_types=['c', 'z']
    )

    config = AutomaticRlTlTrainingConfig(
        dataset=dataset,
        baseline_model=None,
        use_wandb=True,
        results_log=os.path.basename(args.parquet) + f'_improve_{str(args.improve)}_log',
        improve_further=args.improve
    )

    trainer = AutomaticRlTlTraining(config)

    new_model = trainer.train()

    model_path = args.model_path
    save_keras_model(new_model, model_path)



if __name__ == '__main__':
    os.environ['HF_HOME'] = '/cmnfs/proj/bmpc_dlomix/datasets'
    os.environ['HF_DATASETS_CACHE'] = '/cmnfs/proj/bmpc_dlomix/datasets/hf_cache'

    num_proc = 16
    os.environ["OMP_NUM_THREADS"] = f"{num_proc}"
    os.environ["TF_NUM_INTRAOP_THREADS"] = f"{num_proc}"
    os.environ["TF_NUM_INTEROP_THREADS"] = f"{num_proc}"

    tf.config.threading.set_inter_op_parallelism_threads(num_proc)
    tf.config.threading.set_intra_op_parallelism_threads(num_proc)

    main()