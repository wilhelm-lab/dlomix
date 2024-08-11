import os
from oktoberfest.runner import run_job
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
runner = run_job('./configs/refinement_etd_config_2.json')