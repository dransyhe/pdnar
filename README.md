# Primal-Dual Graph Neural Networks for General NP-Hard Combinatorial Optimization


### Installation

1. Create the environment using the `environment.yml` file:

    ```bash
    conda env create -f environment.yml
    ```

2. Activate the environment:

    ```bash
    conda activate pdgnn
    ```

### Usage 

To run the experiments for the three NP-hard problems, use the following commands:

1. **Minimum Vertex Cover (MVC)**

    ```
    python main.py data.algorithm="vertex_cover"
    ```

2. **Minimum Set Cover (MSC)**

    ```
    python main.py data.algorithm="set_cover"
    ```

3. **Minimum Hitting Set (MHS)** 

    ```
    python main.py data.algorithm="hitting_set" model.model.eps=True 
    ```

- Note that  `model.model.eps` controls whether the uniform increase rule in used.

### Optional parameters 
| **Parameter**         | **Description**                                      | **Type**      | **Default Value** | 
|-----------------------|------------------------------------------------------|---------------|-------------------|
| `seed`        | Random seed.               | `int`         | `0`        | 
| `wandb_use`        | Whether to use wandb.                        | `bool`         | `False`        | 
| `inference_only`    | Whether to only perform inference.                          | `bool`         | `False`              | 
| `checkpoint`        | Path of pretrained model for inference only.                            | `str`         | `null`             |                      





