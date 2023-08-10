import os
from alpaca_farm.auto_annotations import alpaca_leaderboard
import csv
from datasets import load_dataset
import json
dataset = load_dataset(path="tatsu-lab/alpaca_farm",name="alpaca_farm_evaluation")["eval"]
instructions = dataset["instruction"]
inputs = dataset["input"]
outputs = []
output_path = "/mnt/petrelfs/chenkeyu1/alapaca_farm_results/output/output-now2.csv"
root_path = "/mnt/petrelfs/chenkeyu1/alapaca_farm_results/now/"
subdirectories = [entry for entry in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, entry))]

with open(output_path, "a", newline="") as csvfile:
    fieldnames = ["name", "score"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    for subdir in subdirectories:
        subdir_path = os.path.join(root_path, subdir)
        name1 = "/alpaca_no_input.json"
        name2 = "/alpaca_with_input.json"
        print(subdir_path)
        data1 = None
        data2=None
        result1=result2=results=result_dict=sorted_result_list=predictions=None
        with open(subdir_path+name1, 'r') as file1:
            data1 = json.load(file1)

        with open(subdir_path+name2, 'r') as file2:
            data2 = json.load(file2)

        result1 = [{"origin_prompt": elem["origin_prompt"], "prediction": elem["prediction"]} for elem in data1.values()]
        result2 = [{"origin_prompt": elem["origin_prompt"], "prediction": elem["prediction"]} for elem in data2.values()]
        results = result1+result2
        for elem in results:
            origin_prompt = elem["origin_prompt"]
            start_marker = "### Instruction:\n"
            end_marker = "\n\n###"

            start_idx = origin_prompt.find(start_marker)
            end_idx = origin_prompt.find(end_marker, start_idx + len(start_marker))
            if start_idx != -1 and end_idx != -1:
                instruction = origin_prompt[start_idx + len(start_marker):end_idx]
                elem["instruction"] = instruction
        result_dict = {elem["instruction"]: elem for elem in results}

        sorted_result_list = [result_dict[elem["instruction"]] for elem in dataset]

        predictions = [elem["prediction"] for elem in sorted_result_list]
        for pred, instruction, input in zip(predictions, instructions, inputs):
            outputs.append({"instruction": instruction, 'input': input, 'output': pred})
        df_results = alpaca_leaderboard(
            path_or_all_outputs=outputs,
            is_add_reference_methods=False,
            annotators_config="annotators/greedy_gpt4/configs.yaml",
        )
        # score = df_results.to_string(float_format="%.2f")
        score = df_results.iloc[0]["win_rate"]
        score = round(score, 2)
        print(score)
        outputs.clear()
        writer.writeheader()
        writer.writerow({"name":subdir_path,"score":score})
