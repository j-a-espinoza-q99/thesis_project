# thesis_project
Mitigating Popularity Bias in LLM-Based Recommender Systems: A Combined Approach of Structured Prompt Engineering and Customized Loss Functions


thesis_project/
├── README.md
├── requirements.txt
├── setup.sh
├── .env.example
├── config/
│   ├── __init__.py
│   ├── config.yaml
│   ├── model_configs.py
│   └── experiment_configs.py
├── data/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── esci_dataset.py
│   └── amazon_c4_dataset.py
├── models/
│   ├── __init__.py
│   ├── blair_model.py
│   ├── deepseek_model.py
│   ├── claude_voyage_model.py
│   ├── custom_model.py
│   ├── loss_functions.py
│   ├── feature_extractors.py
│   └── adapters.py
├── prompts/
│   ├── __init__.py
│   ├── prompt_templates.py
│   ├── llm_prompts.py
│   └── augmenter.py
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py
│   ├── evaluate.py
│   ├── fairness_metrics.py
│   └── benchmark.py
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   ├── train_blair.py
│   ├── train_custom.py
│   └── train_utils.py
├── experiments/
│   ├── run_baseline.sh
│   ├── run_deepseek.sh
│   ├── run_claude.sh
│   ├── run_custom.sh
│   └── run_full_benchmark.sh
├── scripts/
│   ├── download_data.sh
│   ├── prepare_data.py
│   ├── generate_embeddings.py
│   └── eval_search.py
└── utils/
    ├── __init__.py
    ├── helpers.py
    ├── logging_utils.py
    └── gpu_utils.py
