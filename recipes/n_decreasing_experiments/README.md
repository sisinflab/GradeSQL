# N Decreasing Experiments

## Overview

This workflow allows you to efficiently test model performance across different numbers of predictions by:
1. Generating predictions once with the maximum N
2. Reusing cached predictions for smaller N values
3. Automatically creating and running all configurations
4. Collecting results for comparative analysis

## Step 1: Create Initial Recipe

Create your first recipe with the following YAML configuration:

```yaml
n: 32                    # Number of predictions
num_samples_per_batch: 32
n_decreasing: True       # Enable N decreasing functionality
n_step: 1               # Step size for decreasing N
```

**Important:** When `n_decreasing: True` is set, the system will:
- Generate 32 predictions per question initially
- Create subsampled datasets for N decreasing experiments
- Use `n_step` to control the decrement (for `n_step=1`, the n are 32, 31, 30, ..., 1)

## Step 2: Run Initial Experiment

Execute your first experiment with the above configuration to generate the initial N predictions.

## Step 3: Configure Template

After the initial experiment completes:

1. Copy your recipe configuration to `config_template`
2. Modify the following parameters:

```yaml
n: N_MAX                     # This will be replaced automatically
num_samples_per_batch: 32
n_decreasing: False      # Disable for individual runs
n_step: 1
```

## Step 4: Generate Decreasing N Recipes

1. Navigate to `src/scripts`
2. Modify the parameters according to your needs:

```bash
START_N=32  # Starting value of N
STEP=1      # Step size for decreasing N
```

3. Run the recipe generation script:

```bash
sh create_recipes_decreasing_n.sh
```

This will automatically create 32 recipes (in this example) with N values decreasing by `n_step`.

## Step 5: Execute Batch Experiments

Run all experiments with:

```bash
$RUN_BATCH_EXPERIMENTS
```

## Results

After completion, you will find results for each N value in the `results` folder, organized by their respective N.