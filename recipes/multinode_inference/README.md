# Multinode Inference Guide

If you need to process a large dataset or handle a high number of generations per question (N), you can use multinode inference to distribute computation across multiple nodes and then aggregate the results.

## Example Scenario

Let's say you have 2000 questions to process on your dataset and want to split the computation between 10 nodes. Each node will process 200 questions.

## Step-by-Step Setup

### 1. Configure Template Parameters

In your `config_template`, specify these placeholders:

```yaml
start_query: START_QUERY_VALUE
offset_start_query: OFFSET_VALUE
```

### 2. Set Script Parameters

Navigate to `src/scripts` and configure the following parameters in `create_recipes_multinode_inference.sh`:

```bash
OFFSET=200
NUM_FILES=10
```

Then run:

```bash
sh create_recipes_multinode_inference.sh
```

This will create 10 recipes, where each experiment processes 200 questions from the dataset.

### 3. Configure Job Submission

Go to `submit_multinode_inference.sh` and modify these parameters:

```bash
start_recipe=1
end_recipe=10
step=1
```

### 4. Launch All Nodes

To launch all nodes, run:

```bash
sh src/scripts/submit_multinode_inference.sh
```

### 5. Merge Results

After all recipes have completed, go to `src/utils/merge_multinode_preds.py` and modify these parameters to match your setup:

```python
NUM_NODES = 10
NUM_QUERIES_PER_NODE = 200
```

Then run:

```bash
python merge_multinode_preds.py
```

## Results

Once complete, you'll find all predictions merged into a single file in the cache folder. You've successfully parallelized computation with a 10x speedup factor!