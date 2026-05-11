import argparse
import os
import sys
import time
import re

import numpy as np
import mlflow
import pandas as pd
from lamarr_energy_tracker.ground_truth_tracking import GroundTruthTracker
from lamarr_energy_tracker.tracker import EnergyTracker

from util import print_colored_block, get_processor_name, get_gpu_name, save_webcam_image

def read_queries(random=True):
    conversations = pd.read_json(path_or_buf=os.path.join(os.path.dirname(__file__), 'llm_baseline_conversations_puffin.jsonl'), lines=True)
    conversations.set_index('id', inplace=True)
    if random:
        conversations = conversations.sample(frac=1)
    return [conv[0]['value'] for conv in conversations['conversations']]

def parse_param_count(s):
    if isinstance(s, str):
        s = s.strip()
        if s.endswith('B'):
            return float(s[:-1]) * 1e9
        elif s.endswith('M'):
            return float(s[:-1]) * 1e6
        elif s.endswith('K'):
            return float(s[:-1]) * 1e3
        else:
            try:
                return float(s)
            except Exception:
                return s
    return s

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Inference benchmarking with ollama LLMs")
    # data and model input
    parser.add_argument("--model", default="gemma3:1b")
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument("--nogpu", type=int, default=0)
    parser.add_argument("--seconds", type=int, default=900, help="number of seconds to profile model on a subset of the data -- 0 process complete")
    args = parser.parse_args()
    mlflow.log_dict(args.__dict__, 'config.json')

    # log important params
    params = {
        'model': args.model,
        'temperature': args.temperature,
        'dataset': 'LLM Benchmark (Puffin)',
        'task': 'Inference',
        'architecture': get_processor_name() if args.nogpu else get_gpu_name(),
    }
    for key, val in params.items():
        mlflow.log_param(key, val)
    
    # if required, disable gpu
    if args.nogpu:
        os.environ["OLLAMA_NO_GPU"] = "1"
    
    # load data
    queries = read_queries()

    # delete old models (for freeing disk space), load new model and its meta info, init inference (often has some temporal overhead)
    import ollama
    if len(ollama.list().models) > 0:
        if len(ollama.list().models) > 1 or ollama.list().models[0].model != args.model:
            for mod in ollama.list().models:
                ollama.delete(mod.model)
    ollama.pull(args.model)
    resp = ollama.chat(model=args.model, messages=[{"role": "user", "content": f"Can you answer questions?"}])
    mlflow.log_param('file_size', ollama.list().models[0].size)
    mlflow.log_param('parameters', parse_param_count(ollama.list().models[0].details.parameter_size))

    # prepare evaluations
    times, n_samples, tokens = [], 0, {'in': [], 'out': []}
    gt_tracker = GroundTruthTracker(verbose=False)
    gt_tracker.start()
    tracker = EnergyTracker(output_dir=os.getcwd())
    tracker.start()
    save_webcam_image("capture_start.jpg")

    # run evaluations but watch for time limit
    print_colored_block(f'STARTING ENERGY PROFILING FOR   {args.model.upper()}   temperature {args.temperature} on   {"CPU" if args.nogpu else "GPU"}')
    # run inference
    for query in queries:
        t0 = time.time()
        resp = ollama.chat(model=args.model, messages=[{"role": "user", "content": query}], options={"temperature": args.temperature})
        try:
            tokens['in'].append(resp['prompt_eval_count'])
            tokens['out'].append(resp['eval_count'])
        except:
            pass
        times.append(time.time() - t0)
        n_samples += 1
        remaining = args.seconds - sum(times) if args.seconds and len(times) < 5 else args.seconds - (sum(times) + np.average(times))
        print(f"\rProcessed queries: {n_samples} | Remaining time: {remaining:.1f}s", end='', flush=True)
        if args.seconds and remaining < 0:
            break
    print_colored_block(f'STOPPING ENERGY PROFILING FOR   {args.model.upper()}  temperature {args.temperature} on   {"CPU" if args.nogpu else "GPU"}', ok=False)
    eval_gt = gt_tracker.stop()
    tracker.stop(print_summary=False)
    save_webcam_image("capture_stop.jpg")

    # add average amount of tokens if there we any errors:
    tokens['in'] += [np.mean(tokens['in'])] * (n_samples - len(tokens['in']))
    tokens['out'] += [np.mean(tokens['out'])] * (n_samples - len(tokens['in']))

    # aggregate results
    emissions = 'emissions.csv'
    emission_data = pd.read_csv(emissions).to_dict()
    results = {
        'n_tokens_in': sum(tokens['in']),
        'n_tokens_out': sum(tokens['out']),
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
    }

    # log results & cleanup
    mlflow.log_param('n_samples', n_samples)
    for key, val in results.items():
        mlflow.log_metric(key, val)
    for f in [emissions, 'capture_start.jpg', 'capture_stop.jpg']:
        if os.path.isfile(f):
            mlflow.log_artifact(f)
            os.remove(f)
    mlflow.end_run()
    print(results)
    print('n_samples', n_samples)
    sys.exit(0)
