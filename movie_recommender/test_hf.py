from huggingface_hub import list_datasets
import json

datasets = list(list_datasets(search="tmdb", limit=10))
for d in datasets:
    print(d.id)
