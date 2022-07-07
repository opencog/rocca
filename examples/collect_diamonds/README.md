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

The three optimal action plans required to get a reward are:

1. `outside(self, house) ∧ do(go_to_key) ↝ hold(self, key)`

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
2. `outside(self, house) ∧ hold(self, key) ∧ do(go_to_house) ↝ inside(self, house)`

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

3. `inside(self, house) ∧ do(go_to_diamonds) ↝ Reward(1)`

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

## Usage

1. Clone malmo https://github.com/Microsoft/malmo

   Note: No need to compile malmo as a pre-built version (MalmoPython)
   has been included under `../rocca/malmo`. Only install the required
   dependencies to launch minecraft.

2. Launch `<MALMO>/Minecraft/launchClient.sh`.

3. Launch collect diamonds `python collect_diamonds.py`.

  - Note: If there is a need to visualize mined cognitive schematics,
    do the following before step 3.

    - Build and run atompace-explorer here
      https://github.com/tanksha/atomspace-explorer rocca_demo branch.

    - Start the RESTAPI `python start_rest_service.py`

    - Open `collect_diamonds.py` and set `self.visualize_cogscm = True`
