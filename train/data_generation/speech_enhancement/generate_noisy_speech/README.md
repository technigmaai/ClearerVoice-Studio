This script generates training data for speech enhancement models. Once the data generation is complete, you can use the output to retrain or fine-tune your models.

# Usage Instructions
1. Prepare file lists: Provide two file lists: one for clean speech and another for noise. Examples can be found in `data/data_scp/speech.scp` and `data/data_scp/noise.scp`.
2. Configure settings: Adjust the parameters in `config/para.cfg` according to your needs. To test the script, run the demo using `bash run.sh`.
3. Loop generation (optional): Set run_num in `run.sh` to repeat the generation process for multiple outputs.
