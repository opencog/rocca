# Collect diamonds

## Description

Agent for a simple minecraft environment which includes blocks of
diamonds locked inside a house and a key to the house. The agent must
look for the key, open the house and collect a diamond to get a
reward.

## Status

The Rocca agent is able to learn and plan the desired cognitive
schematics leading to getting/accumulating a reward via pattern mining
and temporal deduction.

The following are optimal action plans the agent learned to get a reward:

1. Monoaction plans: 

1.1. `outside(self, house) ∧ do(go_to_key) ↝ hold(self, key)`

```scheme
(BackPredictiveImplicationScopeLink (stv 1 0.0111248)
  (VariableSet
  ) ; [80075fffffff523a][1]
  (SLink
    (ZLink
    ) ; [800fbffffffe8ce4][1]
  ) ; [da5f815ba9d4009f][1]
  (AndLink
    (EvaluationLink (stv 0.64 0.0588235)
      (PredicateNode "outside") ; [72730412e28a734][1]
      (ListLink
        (ConceptNode "self") ; [40b11d11524bd751][1]
        (ConceptNode "house") ; [63eb9919f37daa5f][1]
      ) ; [aadca36fe9d1a468][1]
    ) ; [ca0c329fb1ab493b][1]
    (ExecutionLink
      (SchemaNode "go_to_key") ; [7f46c329a5e57604][1]
    ) ; [f8086e6fdf73cdf4][1]
  ) ; [ddda31153cf2aa6d][1]
  (EvaluationLink (stv 0.32 0.0588235)
    (PredicateNode "hold") ; [4b1b7a8b0a4d2853][1]
    (ListLink
      (ConceptNode "self") ; [40b11d11524bd751][1]
      (ConceptNode "key") ; [4d0844146f96d3][1]
    ) ; [e7a9c95ae7484b28][1]
  ) ; [e6a8f21e6b37d8f0][1]
) ; [ff069058911f0233][1]
```
1.2. `outside(self, house) ∧ hold(self, key) ∧ do(go_to_house) ↝ inside(self, house)`

```scheme
(BackPredictiveImplicationScopeLink (stv 1 0.00621118)
  (VariableSet
  ) ; [80075fffffff523a][1]
  (SLink
    (ZLink
    ) ; [800fbffffffe8ce4][1]
  ) ; [da5f815ba9d4009f][1]
  (AndLink
    (ExecutionLink
      (SchemaNode "go_to_house") ; [7e1737e3e117d059][1]
    ) ; [93427319ec122fff][1]
    (EvaluationLink (stv 0.64 0.0588235)
      (PredicateNode "outside") ; [72730412e28a734][1]
      (ListLink
        (ConceptNode "self") ; [40b11d11524bd751][1]
        (ConceptNode "house") ; [63eb9919f37daa5f][1]
      ) ; [aadca36fe9d1a468][1]
    ) ; [ca0c329fb1ab493b][1]
    (EvaluationLink (stv 0.32 0.0588235)
      (PredicateNode "hold") ; [4b1b7a8b0a4d2853][1]
      (ListLink
        (ConceptNode "self") ; [40b11d11524bd751][1]
        (ConceptNode "key") ; [4d0844146f96d3][1]
      ) ; [e7a9c95ae7484b28][1]
    ) ; [e6a8f21e6b37d8f0][1]
  ) ; [fbf4892ec0643e2c][1]
  (EvaluationLink
    (PredicateNode "inside") ; [63398dcfcf85c8a3][1]
    (ListLink
      (ConceptNode "self") ; [40b11d11524bd751][1]
      (ConceptNode "house") ; [63eb9919f37daa5f][1]
    ) ; [aadca36fe9d1a468][1]
  ) ; [871c182b52e89756][1]
) ; [a2984b519c1ddc4f][1]
```

1.3. `inside(self, house) ∧ do(go_to_diamonds) ↝ Reward(1)`

```scheme
(BackPredictiveImplicationScopeLink (stv 1 0.00621118)
  (VariableSet
  ) ; [80075fffffff523a][1]
  (SLink
    (ZLink
    ) ; [800fbffffffe8ce4][1]
  ) ; [da5f815ba9d4009f][1]
  (AndLink
    (EvaluationLink
      (PredicateNode "inside") ; [63398dcfcf85c8a3][1]
      (ListLink
        (ConceptNode "self") ; [40b11d11524bd751][1]
        (ConceptNode "house") ; [63eb9919f37daa5f][1]
      ) ; [aadca36fe9d1a468][1]
    ) ; [871c182b52e89756][1]
    (ExecutionLink
      (SchemaNode "go_to_diamonds") ; [7aee74cf6bad6442][1]
    ) ; [a2630f96ffbe0861][1]
  ) ; [b4373ec3773f1783][1]
  (EvaluationLink
    (PredicateNode "Reward") ; [155bb4d713db0d51][1]
    (NumberNode "1") ; [2cf0956d543cff8e][1]
  ) ; [d3cee8bdda06ffcb][1]
) ; [8a298365b46b204b][1]
```

2. Polyaction plans

2.1.`(outside(self, house) ∧ do(go_to_key) ⩘ do(go_to_house)) ⩘ do(go_to_diamonds) ↝ Reward(1)`

```scheme
(BackPredictiveImplicationScopeLink (stv 1 0.00503106)
  (VariableSet
  ) ; [80075fffffff523a][3]
  (SLink
    (ZLink
    ) ; [800fbffffffe8ce4][3]
  ) ; [da5f815ba9d4009f][3]
  (BackSequentialAndLink
    (SLink
      (ZLink
      ) ; [800fbffffffe8ce4][3]
    ) ; [da5f815ba9d4009f][3]
    (BackSequentialAndLink
      (SLink
        (ZLink
        ) ; [800fbffffffe8ce4][3]
      ) ; [da5f815ba9d4009f][3]
      (AndLink (stv 0.18 0.0588235)
        (EvaluationLink (stv 0.64 0.0588235)
          (PredicateNode "outside") ; [72730412e28a734][3]
          (ListLink
            (ConceptNode "self") ; [40b11d11524bd751][3]
            (ConceptNode "house") ; [63eb9919f37daa5f][3]
          ) ; [aadca36fe9d1a468][3]
        ) ; [ca0c329fb1ab493b][3]
        (ExecutionLink
          (SchemaNode "go_to_key") ; [7f46c329a5e57604][3]
        ) ; [f8086e6fdf73cdf4][3]
      ) ; [ddda31153cf2aa6d][3]
      (ExecutionLink
        (SchemaNode "go_to_house") ; [7e1737e3e117d059][3]
      ) ; [93427319ec122fff][3]
    ) ; [d526b02a321df7c7][3]
    (ExecutionLink
      (SchemaNode "go_to_diamonds") ; [7aee74cf6bad6442][3]
    ) ; [a2630f96ffbe0861][3]
  ) ; [b0235094e5713353][3]
  (EvaluationLink (stv 0.1 0.0588235)
    (PredicateNode "Reward") ; [155bb4d713db0d51][3]
    (NumberNode "1") ; [2cf0956d543cff8e][3]
  ) ; [d3cee8bdda06ffcb][3]
) ; [bd6c515d92fe0be1][3]
```

2.2. `outside(self, house) ∧ do(go_to_key) ⩘ do(go_to_house) ↝ inside(self, house)`

```scheme
(BackPredictiveImplicationScopeLink (stv 1 0.00559006)
  (VariableSet
  ) ; [80075fffffff523a][3]
  (SLink
    (ZLink
    ) ; [800fbffffffe8ce4][3]
  ) ; [da5f815ba9d4009f][3]
  (BackSequentialAndLink
    (SLink
      (ZLink
      ) ; [800fbffffffe8ce4][3]
    ) ; [da5f815ba9d4009f][3]
    (AndLink (stv 0.18 0.0588235)
      (EvaluationLink (stv 0.64 0.0588235)
        (PredicateNode "outside") ; [72730412e28a734][3]
        (ListLink
          (ConceptNode "self") ; [40b11d11524bd751][3]
          (ConceptNode "house") ; [63eb9919f37daa5f][3]
        ) ; [aadca36fe9d1a468][3]
      ) ; [ca0c329fb1ab493b][3]
      (ExecutionLink
        (SchemaNode "go_to_key") ; [7f46c329a5e57604][3]
      ) ; [f8086e6fdf73cdf4][3]
    ) ; [ddda31153cf2aa6d][3]
    (ExecutionLink
      (SchemaNode "go_to_house") ; [7e1737e3e117d059][3]
    ) ; [93427319ec122fff][3]
  ) ; [d526b02a321df7c7][3]
  (EvaluationLink (stv 0.34 0.0588235)
    (PredicateNode "inside") ; [63398dcfcf85c8a3][3]
    (ListLink
      (ConceptNode "self") ; [40b11d11524bd751][3]
      (ConceptNode "house") ; [63eb9919f37daa5f][3]
    ) ; [aadca36fe9d1a468][3]
  ) ; [871c182b52e89756][3]
) ; [c4a99a11fb34aeb2][3]
```

2.3 `hold(self, key) ∧ do(go_to_house) ⩘ do(go_to_diamonds) ↝ Reward(1)`

```scheme
(BackPredictiveImplicationScopeLink (stv 1 0.00559006)
  (VariableSet
  ) ; [80075fffffff523a][3]
  (SLink
    (ZLink
    ) ; [800fbffffffe8ce4][3]
  ) ; [da5f815ba9d4009f][3]
  (BackSequentialAndLink
    (SLink
      (ZLink
      ) ; [800fbffffffe8ce4][3]
    ) ; [da5f815ba9d4009f][3]
    (AndLink (stv 0.1 0.0588235)
      (ExecutionLink
        (SchemaNode "go_to_house") ; [7e1737e3e117d059][3]
      ) ; [93427319ec122fff][3]
      (EvaluationLink (stv 0.32 0.0588235)
        (PredicateNode "hold") ; [4b1b7a8b0a4d2853][3]
        (ListLink
          (ConceptNode "self") ; [40b11d11524bd751][3]
          (ConceptNode "key") ; [4d0844146f96d3][3]
        ) ; [e7a9c95ae7484b28][3]
      ) ; [e6a8f21e6b37d8f0][3]
    ) ; [876645d6528a6fbb][3]
    (ExecutionLink
      (SchemaNode "go_to_diamonds") ; [7aee74cf6bad6442][3]
    ) ; [a2630f96ffbe0861][3]
  ) ; [c6e23df43eb8fbe6][3]
  (EvaluationLink (stv 0.1 0.0588235)
    (PredicateNode "Reward") ; [155bb4d713db0d51][3]
    (NumberNode "1") ; [2cf0956d543cff8e][3]
  ) ; [d3cee8bdda06ffcb][3]
) ; [f4dde218e5acedc6][3]
```

## Usage

1. Clone Malmo
  ```bash
    git clone https://github.com/Microsoft/malmo
  ```
  NOTE: No need to compile Malmo, as a pre-built version (MalmoPython) is already 
  included in the directory `../rocca/malmo`. All that is required is to install 
  the necessary dependencies ([Windows](https://github.com/microsoft/malmo/blob/master/doc/install_windows.md), [Linux](https://github.com/microsoft/malmo/blob/master/doc/install_linux.md), [MacOSX](https://github.com/microsoft/malmo/blob/master/doc/install_macosx.md)) in order to launch Minecraft.

2. Once the dependencies are installed, launch Minecraft at `port 10000`
  ```bash
    cd Minecraft
    .launchClient.sh -port 10000
 ```

3. Launch the example 
  ```bash
    python collect_diamonds.py
```

Note: If there is a need to visualize mined cognitive schematics using atompace-explorer,
    the following steps are required before launching the example.

    a. Build and run atompace-explorer [here] (https://github.com/tanksha/atomspace-explorer rocca_demo branch).
    b. Start the RESTAPI client (which creates an atomspace to store cognitive schematics to be visualized) 
    ```bash 
      python start_rest_service.py
    ```
    c. Set `self.visualize_cogscm = True` in `collect_diamonds.py`
