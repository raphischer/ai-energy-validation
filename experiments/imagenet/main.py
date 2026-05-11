import argparse
import os
import sys
import time
import multiprocessing as mp
mp.set_start_method("spawn", force=True) # make sure that tensorflow works with mlflow and multiprocessing

import numpy as np
import mlflow
import pandas as pd

from lamarr_energy_tracker.ground_truth_tracking import GroundTruthTracker
from lamarr_energy_tracker.tracker import EnergyTracker

from util import print_colored_block, save_webcam_image
from batch_sizes import lookup_batch_size, find_ideal_batch_size

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Inference benchmarking with keras models on ImageNet")
    # data and model input
    parser.add_argument("--model", default="EfficientNetB0")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--datadir", default="/data/d1/fischer_diss/imagenet")
    parser.add_argument("--batchsize", default=16)
    parser.add_argument("--seconds", type=int, default=120, help="number of seconds to profile model on a subset of the data -- 0 process complete")
    args = parser.parse_args()
    mlflow.log_dict(args.__dict__, 'config.json')
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_gpu = os.environ['CUDA_VISIBLE_DEVICES']

    # identify batch_size and load data
    if args.batchsize:
        batch_size = int(args.batchsize)
    else:
        batch_size = lookup_batch_size(args.model) or find_ideal_batch_size(args.model, args.datadir)
    from data_and_model_loading import load_data_and_model # import also inits tensorflow, so only import now
    model, ds, meta = load_data_and_model(args.datadir, args.model, batch_size=batch_size)
    print('Loaded model')
    meta['dataset'], meta['task'] = 'ImageNet (ILSVRC 2012)', 'Inference'
    for key, val in meta.items():
        mlflow.log_param(key, val)
    model.evaluate(ds.take(2)) # init inference (often has some temporal overhead)
    n_samples = 50000
    print('Initiliazed inference')

    # given limit for evaluation, so only take a small subset of the data
    if args.seconds:
        t0 = time.time()
        model.evaluate(ds.take(1)) # first test a single batch
        t1 = time.time()
        test_n = max(5, np.round(args.seconds / (4 * (t1-t0)))) # test min of five batches, but if very fast, 1/4 of the time limit
        model.evaluate(ds.take(test_n + 1))
        t_single_without_init = (time.time() - t1 - (t1 - t0)) / test_n # remove overhead and calc per sample time
        n_batches = np.round(args.seconds / t_single_without_init)
        n_samples = n_batches * meta['batch_size']
        while n_batches > len(ds): # for very fast models, we maybe need to repeat the dataset several times
            ds = ds.concatenate(ds)
        ds = ds.take(n_batches)
        print(f'Processing {n_batches} batches, per batch expected runtime {t_single_without_init:.4f}s, len ds {len(ds)}')

    # run evaluations while tracking resource consumption
    mlflow.log_param('n_samples', n_samples)
    gt_tracker = GroundTruthTracker(verbose=False)
    tracker = EnergyTracker(output_dir=os.getcwd())
    gt_tracker.start()
    tracker.start()
    save_webcam_image("capture_start.jpg")
    print_colored_block(f'STARTING ENERGY PROFILING FOR  {args.model.upper()}  batch size {args.batchsize} on gpu {use_gpu}')
    eval_res = model.evaluate(ds, return_dict=True) # evaluate on samples
    print_colored_block(f'STOPPING ENERGY PROFILING FOR  {args.model.upper()}  batch size {args.batchsize} on gpu {use_gpu}', ok=False)
    eval_gt = gt_tracker.stop()
    tracker.stop(print_summary=False)
    save_webcam_image("capture_stop.jpg")

    if not args.seconds:
        # evaluate robustness
        _, corr, _ = load_data_and_model(args.datadir, args.model, variant='corrupted_sample', batch_size=meta["batch_size"])
        corr_res = model.evaluate(corr, return_dict=True)
        for key, val in corr_res.items():
            eval_res[f'corr_{key}'] = val

    # aggregate results
    emissions = 'emissions.csv'
    emission_data = pd.read_csv('emissions.csv').to_dict()
    eval_res.update({
    # codecarbon logs
        'running_time_total': emission_data['duration'][0],
        'running_time':  emission_data['duration'][0] / n_samples,
        'power_draw_total': emission_data['energy_consumed'][0] * 3.6e6,
        'power_draw': emission_data['energy_consumed'][0] * 3.6e6 / n_samples,
        # ground-truth logs
        'power_draw_total_gt': eval_gt['energy_consumed'] * 3.6e6,
        'power_draw_gt': eval_gt['energy_consumed'] * 3.6e6 / n_samples,
        'running_time_total_gt': eval_gt['duration'],
        'running_time_gt':  eval_gt['duration'] / n_samples
    })

    # log results & cleanup
    for key, val in eval_res.items():
        mlflow.log_metric(key, val)
    for f in [emissions, 'capture_start.jpg', 'capture_stop.jpg']:
        if os.path.isfile(f):
            mlflow.log_artifact(f)
            os.remove(f)
    mlflow.end_run()
    print(eval_res)
    sys.exit(0)
