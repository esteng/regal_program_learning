import numpy as np
from pathlib import Path
import json 
import pdb 

import sys
sys.path.insert(0, "third_party/Faithful-COT/source")
from evaluate.evaluate_answer_acc import is_correct, extract_gold_answer, extract_pred_answer

np.random.seed(12)
# for each dataset, split into train/dev/test

outpath = Path("python_data")

for dataset_path in Path("third_party/Faithful-COT/data/").glob('*'): 
    if not dataset_path.is_dir():
        continue
    dataset_name = dataset_path.name
    output_path = f"third_party/Faithful-COT/output_dir/{dataset_name}/test"
    data  = [json.loads(x) for x in open(dataset_path / 'test.jsonl').readlines()]


    for model_path in Path(output_path).glob("*"): 
        model_name = model_path.name


        outputs = [json.loads(x) for x in open(model_path /  'predictions.jsonl').readlines()]
        assert len(data) == len(outputs)

        # only keep examples where model was right 
        data_to_keep, outputs_to_keep = [], []

        for i, (d, o) in enumerate(zip(data, outputs)):
            try:
                gold_answer = extract_gold_answer(dataset_name, d['answer'])
                pred_answer = extract_pred_answer(dataset_name, o['answer'])
                d['id'] = str(d['id'])
                o['id'] = str(o['id'])

                if is_correct(dataset_name, gold_answer, pred_answer): 
                    assert(d['id'] == o['id'])
                    data_to_keep.append((i,d))
                    outputs_to_keep.append((i, o))
            except AssertionError:
                pdb.set_trace()
                continue
            #     print(f"error for {dataset_name}, {model_name}")
            #     continue

        correct_only_idxs = [i for i in range(len(data_to_keep))]
        np.random.shuffle(correct_only_idxs)
        correct_only_idxs = [i for i in correct_only_idxs]

        for idx in correct_only_idxs:
            assert(data_to_keep[idx][1]['id'] == outputs_to_keep[idx][1]['id'])

        if len(correct_only_idxs) < 500:
            train_len = int(0.25 * len(correct_only_idxs))
            dev_len = int(0.25 * len(correct_only_idxs))
            test_len = int(0.5 * len(correct_only_idxs))
            train_idxs = correct_only_idxs[0:train_len]
            dev_idxs = correct_only_idxs[train_len:dev_len + train_len]
            test_idxs = correct_only_idxs[dev_len+train_len : ]
            test_small_idxs = []
        else:
            train_idxs = correct_only_idxs[0:200]
            dev_idxs = correct_only_idxs[200:300]
            test_small_idxs = correct_only_idxs[300:500]
            test_idxs = correct_only_idxs[500:]

        
        # get correct-only data 
        train_data = [(data_to_keep[i], outputs_to_keep[i]) \
                        for i in train_idxs]
        dev_data = [(data_to_keep[i], outputs_to_keep[i]) \
                        for i in dev_idxs]
        test_small_data = [(data_to_keep[i], outputs_to_keep[i]) \
                        for i in test_small_idxs]
        test_data = [(data_to_keep[i], outputs_to_keep[i]) \
                        for i in test_idxs]
        

        # get data also with incorrect for dev and test 
        correct_only_orig_idxs = [x[0] for x in train_data + dev_data + test_small_data + test_data]
        incorrect_also_idxs = [i for i in range(len(data)) if i not in correct_only_idxs]
        dev_correct_only_idxs = [x[0][0] for x in dev_data]
        test_correct_only_idxs = [x[0][0] for x in test_data]
        # data is correct-only data + half of incorrect data (none of the incorrect data is part of train)
        np.random.shuffle(incorrect_also_idxs)
        dev_incorrect_idxs = incorrect_also_idxs[0:int(0.5 * len(incorrect_also_idxs))]
        test_incorrect_idxs = incorrect_also_idxs[int(0.5 * len(incorrect_also_idxs)):]

        # add the two splits together
        dev_with_incorrect_data = [((i, data[i]), (i, outputs[i])) for i in dev_incorrect_idxs + dev_correct_only_idxs]
        test_with_incorrect_data = [((i, data[i]), (i, outputs[i])) for i in test_incorrect_idxs + test_correct_only_idxs]

        Path(outpath / dataset_name / model_name).mkdir(parents=True, exist_ok=True)


        with open(outpath / dataset_name / model_name/ "train_data.jsonl", "w") as f1, \
            open(outpath / dataset_name /  model_name / "train_outputs.jsonl", "w") as f2, \
            open(outpath / dataset_name / model_name / "train_combined.jsonl", "w") as f3:
            for (_, t), (_, o) in train_data: 

                assert(o['id'] == t['id'])
                f1.write(json.dumps(t) + '\n')
                f2.write(json.dumps(o) + '\n') 
    
                combo_dict = {"language": [t['question']], "program": o['completion']} 
                f3.write(json.dumps(combo_dict) + '\n')

        with open(outpath / dataset_name / model_name/ "dev_data.jsonl", "w") as f1, \
            open(outpath / dataset_name /  model_name / "dev_outputs.jsonl", "w") as f2, \
            open(outpath / dataset_name / model_name / "dev_combined.jsonl", "w") as f3:
            for (_, t), (_, o) in dev_data: 
                assert(o['id'] == t['id'])
                f1.write(json.dumps(t) + '\n')
                f2.write(json.dumps(o) + '\n')
                combo_dict = {"language": [t['question']], "program": o['completion']}
                f3.write(json.dumps(combo_dict) + '\n') 

        with open(outpath / dataset_name /  model_name/"test_small_data.jsonl", "w") as f1, \
            open(outpath / dataset_name /  model_name / "test_small_outputs.jsonl", "w") as f2, \
            open(outpath / dataset_name / model_name / "test_small_combined.jsonl", "w") as f3:
            for (_, t), (_, o) in test_small_data: 
                assert(o['id'] == t['id'])
                f1.write(json.dumps(t) + '\n')
                f2.write(json.dumps(o) + '\n')
                combo_dict = {"language": [t['question']], "program": o['completion']}
                f3.write(json.dumps(combo_dict) + '\n')
                

        with open(outpath / dataset_name / model_name/ "test_data.jsonl", "w") as f1, \
            open(outpath / dataset_name /  model_name / "test_outputs.jsonl", "w") as f2:
            for (_, t), (_, o) in test_data: 
                assert(o['id'] == t['id'])
                f1.write(json.dumps(t) + '\n')
                f2.write(json.dumps(o) + '\n')

        
        # don't write programs here since they are not always correct 
        with open(outpath / dataset_name / model_name / "dev_w_incorrect_data.jsonl", "w") as f1:
            for (_, t), (_, o) in dev_with_incorrect_data: 
                try:
                    assert(str(o['id']) == str(t['id']))
                except AssertionError:
                    pdb.set_trace() 
                f1.write(json.dumps(t) + '\n')


        with open(outpath / dataset_name / model_name / "test_w_incorrect_data.jsonl", "w") as f2:
            for (_, t), (_, o) in test_with_incorrect_data: 
                try:
                    assert(str(o['id']) == str(t['id']))
                except AssertionError:
                    pdb.set_trace() 
                f2.write(json.dumps(t) + '\n')
