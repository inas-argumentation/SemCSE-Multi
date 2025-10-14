This is the official code repository for the paper:
### SemCSE-Multi: Multifaceted and Decodable Embeddings for Aspect-Specific and Interpretable Scientific Domain Mapping

We develop a pipeline for training multifaceted embedding models in the scientific domain. This means, that the embedding model outputs multiple embeddings that encode different aspects of the underlying scientific text.
In our experiments, we trained two models, one for the domain of invasion biology and one for the medical domain.
You can use the models like this:

```
from transformers import AutoTokenizer, AutoModel

# Invasion biology model
model = AutoModel.from_pretrained("CLAUSE-Bielefeld/SemCSE-Multi-Invasion-Biology", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("CLAUSE-Bielefeld/SemCSE-Multi-Invasion-Biology")

text = "This is a scientific abstract from the domain of invasion biology."
batch = tokenizer([text], return_tensors='pt')

# Get the embedding for the "species" aspect. Other options are: "hypothesis", "ecosystem", "researchquestion", "methodology" and "recommendation".
output = model(**batch)["species"]
```

```
from transformers import AutoTokenizer, AutoModel

# medical model
model = AutoModel.from_pretrained("CLAUSE-Bielefeld/SemCSE-Multi-Medical", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("CLAUSE-Bielefeld/SemCSE-Multi-Medical")

text = "This is a scientific abstract from the medical domain."
batch = tokenizer([text], return_tensors='pt')

# Get the embedding for the "disease" aspect. Other options are: "patient_group" and "methodology".
output = model(**batch)["disease"]
```

## Performing Experiments and Evaluations

This repository contains all necessary files for running our experiments and evaluations. This includes:
* All data that was used: Scientific abstracts (with labels), generated summaries that were used for training, and generated pariwise assessments that were used for evaluation.
* The code for generating summaries and pairwise assessments
* The code for training all models
* The code for running all evaluations

Each domain (invasion biology/medicine) has its own directory. The filenames should be self-explanatory. To see how to run the individual components, see `main.py` and comment out all unwanted steps.
