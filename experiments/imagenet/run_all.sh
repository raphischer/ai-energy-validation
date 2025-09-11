#! /bin/bash

# Get current timestamp for experiment naming
printf -v now '%(%F_%H-%M-%S)T' -1
exp_name="imagenet_$1_$now"
exp_create_str=$(mlflow experiments create -n $exp_name)
echo $exp_create_str
exp_id=$(echo $exp_create_str | awk '{print $NF}')

# Define model architectures to iterate over
models=("DenseNet121" "DenseNet201" "ResNet50" "ResNet101" "ResNet152" "ResNet50V2" "ResNet101V2" "ResNet152V2" 
        "InceptionResNetV2" "InceptionV3" "MobileNet" "MobileNetV2" "MobileNetV3Large" "MobileNetV3Small"
        "EfficientNetB0" "EfficientNetB2" "EfficientNetB3" "EfficientNetB4"
        "EfficientNetV2B0" "EfficientNetV2B1" "EfficientNetV2B2" "EfficientNetV2B3" "EfficientNetV2S" 
        "NASNetMobile" "Xception" "VGG16" "VGG19" "ConvNeXtBase" "ConvNeXtSmall" "ConvNeXtTiny")

# Loop over all configurations (gpu use, batch size, model)
for g in "0" "-1" # maybe also -1
do
    export CUDA_VISIBLE_DEVICES="$g"
    for b in "4" "16"
    do
        for m in "${models[@]}"
        do
            # Keep trying the mlflow run until it succeeds
            while true
            do
                echo "Running model $m on GPU $g ..."
                timeout $(( $1 * 2 )) mlflow run --experiment-name=$exp_name -e main.py -P model=$m -P datadir=/data/d1/fischer_diss/imagenet -P seconds=$1 -P batchsize=$b ./experiments/imagenet
                
                # Check if the mlflow run succeeded (exit status 0)
                if [ $? -eq 0 ]; then
                    echo "Run succeeded for model $m on GPU $g"
                    break  # Exit the loop if succeeded
                else
                    echo "Run failed for model $m on GPU $g, retrying..."
                    sleep 5
                fi
            done
            # Save experiment data to CSV
            mlflow experiments csv -x $exp_id > "results/$exp_name.csv"
        done
    done
done
