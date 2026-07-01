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

def read_queries(random=42):
    conversations = pd.read_json(path_or_buf=os.path.join(os.path.dirname(__file__), 'llm_baseline_conversations_puffin.jsonl'), lines=True)
    conversations.set_index('id', inplace=True)
    if random:
        conversations = conversations.sample(frac=1, random_state=random)
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
    parser.add_argument("--model", default="gemma3:1b", help="ollama model to benchmark")
    parser.add_argument('--temperature', type=float, default=0.7, help='temperature for sampling, default 0.7')
    parser.add_argument("--nogpu", type=int, default=0, help="disable gpu usage for inference, set to 1 to disable")
    parser.add_argument("--random", type=int, default=42, help="random seed for shuffling the queries, set to 0 to disable shuffling")
    parser.add_argument("--timeout", type=int, default=120, help="timeout for each inference call in seconds")
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
    queries = read_queries(random=args.random)

    # delete old models (for freeing disk space), load new model and its meta info, init inference (often has some temporal overhead)
    import ollama
    if len(ollama.list().models) > 0:
        if len(ollama.list().models) > 1 or ollama.list().models[0].model != args.model:
            for mod in ollama.list().models:
                ollama.delete(mod.model)
    ollama.pull(args.model)
    client = ollama.Client(host='http://localhost:11434', timeout=args.timeout)
    resp = client.chat(model=args.model, messages=[{"role": "user", "content": f"Can you answer questions?"}])
    mlflow.log_param('file_size', ollama.list().models[0].size)
    mlflow.log_param('parameters', parse_param_count(ollama.list().models[0].details.parameter_size))

    # prepare evaluations
    times, failed_times, tokens = [], [], {'in': [0], 'out': [0]}
    gt_tracker = GroundTruthTracker(verbose=False, crash_if_unavailable=False) # track ground truth energy and runtime (if available)
    tracker = EnergyTracker(output_dir=os.getcwd())
    gt_tracker.start()
    tracker.start()
    save_webcam_image("capture_start.jpg")

    # run evaluations but watch for time limit
    print_colored_block(f'STARTING ENERGY PROFILING FOR   {args.model.upper()}   temperature {args.temperature} on   {"CPU" if args.nogpu else "GPU"}')
    # run inference
    for query in queries:
        t0 = time.time()
        try:
            resp = client.chat(model=args.model, messages=[{"role": "user", "content": query}], options={"temperature": args.temperature})
            tokens['in'].append(resp['prompt_eval_count'])
            tokens['out'].append(resp['eval_count'])
        except Exception as e:
            failed_times.append(time.time() - t0)
        times.append(time.time() - t0)
        remaining = args.seconds - (sum(times) + np.average(times))
        print(f"\rProcessed queries: {len(times):<3} | Remaining time: {remaining:.1f}s | mean time: {np.average(times):.2f}s | std time: {np.std(times):.2f}s  | max time: {np.max(times):.2f}s | errors: {len(failed_times)}", end='', flush=True)
        if args.seconds and remaining < 0:
            break
    print_colored_block(f'STOPPING ENERGY PROFILING FOR  {args.model.upper()}  temperature {args.temperature} on   {"CPU" if args.nogpu else "GPU"}', ok=False)
    eval_gt = gt_tracker.stop()
    tracker.stop(print_summary=False)
    save_webcam_image("capture_stop.jpg")

    print(f"\rProcessed queries: {len(times):<3} | Remaining time: {remaining:.1f}s | mean time: {np.average(times):.2f}s | std time: {np.std(times):.2f}s  | max time: {np.max(times):.2f}s | errors: {len(failed_times)}")

    if failed_times: # add estimates for untracked token counts if there we any timeouts:
        time_with_tokens = sum(times) - sum(failed_times)
        time_without_tokens = sum(failed_times)
        in_tokens_per_s = sum(tokens['in']) / time_with_tokens
        out_tokens_per_s = sum(tokens['out']) / time_with_tokens
        tokens['in'].append(in_tokens_per_s * time_without_tokens)
        tokens['out'].append(out_tokens_per_s * time_without_tokens)

    # aggregate results
    emissions = 'emissions.csv'
    emission_data = pd.read_csv(emissions).to_dict()
    results = {
        'n_tokens_in': sum(tokens['in']),
        'n_tokens_out': sum(tokens['out']),
        'avg_time': np.average(times),
        'max_time': np.max(times),
        'std_time': np.std(times),
        # codecarbon logs
        'running_time_total': emission_data['duration'][0],
        'running_time':  emission_data['duration'][0] / sum(tokens['out']),
        'power_draw_total': emission_data['energy_consumed'][0] * 3.6e6,
        'power_draw': emission_data['energy_consumed'][0] * 3.6e6 / sum(tokens['out']),
        # ground-truth logs
        'power_draw_total_gt': eval_gt['energy_consumed'] * 3.6e6,
        'power_draw_gt': eval_gt['energy_consumed'] * 3.6e6 / sum(tokens['out']),
        'running_time_total_gt': eval_gt['duration'],
        'running_time_gt':  eval_gt['duration'] / sum(tokens['out'])
    }

    # log results & cleanup
    mlflow.log_param('n_samples', len(times))
    mlflow.log_param('n_errors', len(failed_times))
    for key, val in results.items():
        mlflow.log_metric(key, val)
    for f in [emissions, 'capture_start.jpg', 'capture_stop.jpg']:
        if os.path.isfile(f):
            mlflow.log_artifact(f)
            os.remove(f)
    mlflow.end_run()
    print(results)
    sys.exit(0)
