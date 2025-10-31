import json
import os

def prepare_t5_data(input_file="data/synthetic_examples.jsonl", output_file="data/t5_train_data.jsonl"):
    """
    Converts the raw JSONL data into the input/target format for T5 finetuning.
    T5 Input: "normalize claim: <raw_text>"
    T5 Target: <JSON_string_without_raw_text_field>
    """
    print(f"Loading data from {input_file}...")
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    # Split data (90% train, 10% validation)
    train_size = int(0.9 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]

    print(f"Total: {len(data)}, Train: {len(train_data)}, Val: {len(val_data)}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Process and save for Hugging Face `datasets` format
    for split_data, suffix in zip([train_data, val_data], ["train", "val"]):
        output_path = f"data/t5_data_{suffix}.jsonl"
        print(f"Writing {suffix} data to {output_path}")
        with open(output_path, 'w') as out_f:
            for item in split_data:
                # 1. Create T5 Input
                input_text = f"normalize claim: {item['raw_text']}"
                
                # 2. Create T5 Target (Must be a clean JSON string, remove the raw_text field)
                target_json = item['structured_data'].copy()
                target_json.pop('raw_text', None) 
                target_text = json.dumps(target_json, separators=(',', ':')) # Compact JSON string

                output_obj = {"input_text": input_text, "target_text": target_text}
                out_f.write(json.dumps(output_obj) + '\n')

    print("Data preparation complete.")

if __name__ == "__main__":
    prepare_t5_data()