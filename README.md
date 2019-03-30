# pytorch-CPR

## Requirements



## Training

```bash
cd ./exp/{your_experiment_folder}
vim {your_experiment_name}.json     # follow the format in default json
cd {project_root}/src/
python main.py -c ../exp/{your_experiment_folder}/{your_experiment_name}.json
```

## Testing

```bash
cd {project_root}/src/
python eval.py -c ../exp/{your_experiment_folder}/{your_experiment_name}.json
```
