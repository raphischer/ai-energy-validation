#! /bin/bash

# Get current timestamp for experiment naming
printf -v now '%(%F_%H-%M-%S)T' -1
exp_name="ollama_$1_$now"
exp_create_str=$(mlflow experiments create -n $exp_name)
echo $exp_create_str
exp_id=$(echo $exp_create_str | awk '{print $NF}')

# Define model architectures to iterate over
models=("gpt-oss:20b" "deepseek-r1:8b" "deepseek-r1:1.5b"
        "gemma3:4b" "gemma3:270m" "gemma3n:e4b"
        "qwen3:8b" "qwen3:4b" "qwen3:0.6b"
        "llama3.1:8b" "llama3.2:3b" "tinyllama:1.1b"
        "mistral:7b" "mistral-small3.2:24b" "magistral:24b"
        "phi3:3.8b" "phi4:14b" "phi4-reasoning:14b"
        "dolphin3:8b" "olmo2:7b")

# Loop over all models and GPUs
for m in "${models[@]}"
do
    for t in "0.1" "0.7"
    do
        # Keep trying the mlflow run until it succeeds
        while true
        do
            echo "Running model $m on GPU $g ..."
            timeout $(( $1 * 3 )) mlflow run --experiment-name=$exp_name -e main.py -P model=$m -P seconds=$1 -P temperature=$t ./experiments/ollama
            
            # Check if the mlflow run succeeded (exit status 0)
            if [ $? -eq 0 ]; then
                echo "Run succeeded for model $m on GPU $g"
                break  # Exit the loop if succeeded
            else
                echo "Run failed for model $m on GPU $g, retrying..."
            fi
        done

        # Save experiment data to CSV
        mlflow experiments csv -x $exp_id > "results/$exp_name.csv"
    done
done
