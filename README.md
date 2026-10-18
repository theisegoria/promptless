# Promptless
Python CLI plugin/app designed to replace streaming prompt chains that assist conversational agents in real time. It sets up queues of boolean indicators and flaggers relying on contextual information within the encoded texts’ vector embeddings.

### Requirements
- `sentence_transformers`
- `numpy`

### Alternative Approach
- Build around huggingface `transformers` chatting pipeline but I am tired of reading docs.

### Extension or TODO
- [ ] Add queue to CLI
- [x] Add Observations obj - Certainity metrics to enable self-adjustment to unseen data. Although, for the case of assisting conversational agents, this task seems fairly straightforward. For more complainings, refer to my post [here](https://mimiphanblog.wordpress.com/2025/02/23/stop-button/). Thank chu.

### Demo usage
- To set up StopButton Agent
```shell
    bash ./setup.sh
```

- Loading Pipeline Example: StopButton
```python
from promptless import Ember
pipe = Ember.load("StopButton")
```
or with CLI
```shell
    python promptless load --name StopButton --action predict --inputs hello hru
```
**Viewing target corpus stored as `pos` attribute**

Example on StopButton's target corpus.

```python
# Viewing the stored target embeddings
pipe.pos
```




    State(text=['Goodbye! Take care!', 'Farewell, my friend.', 'See you soon!', 'Until we meet again!', 'Wishing you all the best!', 'Stay safe and take care!', 'It was great knowing you!', 'Good luck on your journey!', 'Parting is such sweet sorrow.', 'Catch you later!', 'Bye for now!', 'May our paths cross again!', 'Take care and keep in touch!', 'So long, and thanks for everything!', 'Adieu, until next time!', 'Keep shining! Farewell!', "I'll miss you! Stay well.", 'Time to say goodbye. Be happy!', 'See you in another life, maybe!', 'The end of one journey is the start of another!', '[BYE]'], x=array([[ 0.03076916,  0.0126105 ,  0.05677234, ..., -0.05001175,
            -0.02077309,  0.01577766],
           [ 0.0134427 ,  0.11560822,  0.04732521, ...,  0.05116416,
            -0.03377761,  0.03247534],
           [-0.0577333 , -0.0587927 ,  0.02223006, ...,  0.03580647,
            -0.02317192,  0.00483897],
           ...,
           [-0.0690318 , -0.11017422,  0.04574804, ..., -0.09984987,
            -0.0621111 , -0.00965871],
           [ 0.00451478, -0.05803003,  0.02980193, ..., -0.03991203,
            -0.00696008, -0.021196  ],
           [ 0.03393115,  0.05990824,  0.02432098, ...,  0.04659498,
             0.10383081, -0.00313707]], shape=(21, 384), dtype=float32))


**Example use case during streaming**

For boolean outputs:

```python
# Testing Prediction func (returns boolean True = Target, False = otherwise)

test_example = "BYE"
pipe.predict(test_example).item()  # raw return: np.True_ ...
```




    True



For Probability scores:

```python
# Testing proba scores
pipe.predict_proba(test_example)
```




    array([0.58961064, 0.4103894 ], dtype=float32)
