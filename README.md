# MindGames

Code associated to the MindGames article
We generate epistemic reasoning problems (akin to muddy children or drinking logicians) using modal logic to target theory of mind (tom) in natural language processing models.

🤗 https://hf.co/datasets/sileod/mindgames

Published at EMNLP Findings 2023
```bib
@inproceedings{sileo-lernould-2023-mindgames,
    title = "{M}ind{G}ames: Targeting Theory of Mind in Large Language Models with Dynamic Epistemic Modal Logic",
    author = "Sileo, Damien  and
      Lernould, Antoine",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.303/",
    doi = "10.18653/v1/2023.findings-emnlp.303",
    pages = "4570--4577",
    abstract = "Theory of Mind (ToM) is a critical component of intelligence but its assessment remains the subject of heated debates. Prior research applied human ToM assessments to natural language processing models using either human-created standardized tests or rule-based templates. However, these methods primarily focus on simplistic reasoning and require further validation. Here, we leverage dynamic epistemic logic to isolate a particular component of ToM and to generate controlled problems. We also introduce new verbalization techniques to express these problems in English natural language. Our findings indicate that some language model scaling (from 70M to 6B and 350M to 174B) does not consistently yield results better than random chance. While GPT-4 demonstrates superior epistemic reasoning capabilities, there is still room for improvement. Our code and datasets are publicly available."
}
```

Contact (interested in logic+NLP colabs) damien.sileo@inria.fr
