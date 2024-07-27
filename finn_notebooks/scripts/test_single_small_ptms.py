import os
import tensorflow as tf
import argparse
import sys
from pyarrow.parquet import ParquetFile
sys.path.append('../../bmpc_shared_scripts/refinement_transfer_learning')
sys.path.append('../../bmpc_shared_scripts/oktoberfest_interface')
sys.path.append('../../bmpc_shared_scripts/prepare_dataset')
from get_updated_alphabet import get_modification
from oktoberfest_interface import load_keras_model, process_dataset
from automatic_rl_tl import AutomaticRlTlTraining, AutomaticRlTlTrainingConfig
from dlomix.data.dataset import load_processed_dataset


def main():
    parser = argparse.ArgumentParser(prog='Test single small PTMs')
    parser.add_argument('--parquet', type=str, required=True)

    args = parser.parse_args()

    file = ParquetFile(args.parquet)
    modifications = set()
    for batch in file.iter_batches():
        for cur_seq in batch['modified_sequence']:
            cur_mods = get_modification(str(cur_seq))
            modifications |= set(cur_mods)
    
    model = load_keras_model('/cmnfs/proj/bmpc_dlomix/models/cid_hcd_only_models/noptm_baseline_full_bs1024_unmod_extended_cid/8a1af0c5-e446-4113-a570-5a4066af7f62.keras')
    dataset = process_dataset(
        parquet_file_path=args.parquet,
        model=model,
        modifications=list(modifications),
        ion_types=['y', 'b']
    )

    config = AutomaticRlTlTrainingConfig(
        dataset=dataset,
        baseline_model=model,
        use_wandb=True
    )

    trainer = AutomaticRlTlTraining(config)

    new_model = trainer.train()


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