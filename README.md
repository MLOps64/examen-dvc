# Examen DVC et Dagshub
Dans ce dépôt vous trouverez l'architecture proposé pour mettre en place la solution de l'examen. 

```bash       
├── examen_dvc          
│   ├── data       
│   │   ├── processed      
│   │   └── raw       
│   ├── metrics       
│   ├── models      
│   │   ├── data      
│   │   └── models        
│   ├── src       
│   └── README.md.py       
```
N'hésitez pas à rajouter les dossiers ou les fichiers qui vous semblent pertinents.

Vous devez dans un premier temps *Fork* le repo et puis le cloner pour travailler dessus. Le rendu de cet examen sera le lien vers votre dépôt sur DagsHub. Faites attention à bien mettre https://dagshub.com/licence.pedago en tant que colaborateur avec des droits de lecture seulement pour que ce soit corrigé.

Vous pouvez télécharger les données à travers le lien suivant : https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv.


#                               EXAMEN
## INSTALLATION
### Clonage repo github
```
~/workspace/sprint3/DVC$ git clone https://github.com/MLOps64/examen-dvc.git
```
### Virtual env
```
cd examen-dvc/
python3 -m venv venv
source venv/bin/activate
```
### Import raw file
```
cd ./data/raw
wget https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv

```
- On ajoute dans ./data/.gitignore :
    - raw
    - processed
### Installation DVC
```
pip3 install dvc
dvc init
dvc config core.analytics false
```
### Integration Dagshub
- Create project from github
- add remote s3 
```
dvc remote add origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/e.papet/examen-dvc.s3
dvc remote modify origin --local access_key_id 538c70be81609dd3be8b5a773abc3a78c1a7c1a0
dvc remote modify origin --local secret_access_key 538c70be81609dd3be8b5a773abc3a78c1a7c1a0
dvc add data/raw
dvc add data/processed
dvc remote default origin
pip install "dvc[s3]"
dvc commit
dvc push
```
### First slpit
- add make_dataset.py
```
/bin/python3 /home/ubuntu/workspace/sprint3/DVC/examen-dvc/src/data/make_dataset.py
dvc commit
git add data/processed.dvc requirements.txt src/data/__init__.py src/data/make_dataset.py
git commit -a -m "First Slipt Data"
git push
dvc push
```
### Normalisation des données (les données sont dans des échelles très variés donc une normalisation est nécessaire)
- On drop la colonne 'date'
- On va utilister la methode Distribution Normale (https://inside-machinelearning.com/normaliser-donnees-2-minutes/#boxzilla-12146)
- Methode StandardScaler() de scikit-learn
```
from sklearn import preprocessing
transformer = preprocessing.StandardScaler().fit(df_features[])
```

### GridSearch
- L'analyse prédictive tourne généralement autour de la classification (catégorisation des données) ou de la régression (prévision de valeurs continues) :
    - Classification : Catégorisation des données lorsque la variable de résultat appartient à un ensemble prédéfini (par exemple, 'Oui' ou 'Non').
    - Régression : Prévision de valeurs continues lorsque la variable de résultat est réelle ou continue (par exemple, poids ou prix).
- Ici on va utiliser la régression linéaire pour prévoir la concentration
- On va tester le randon et le grid
- production de lr_grid_search_estimator.pkl et lr_randon_search_estimartor. sauvegardé dans models
- dvc add models/lr_grid_search_estimator.pkl models/lr_randon_search_estimartor
- Les hyperparametres trouvés par les 2 méthodes sont les mêmes
```
2025-02-26 15:24:40,409 - root - INFO - => GridSearchCV :best estimator : LinearRegression(n_jobs=1) - best param : {'copy_X': True, 'fit_intercept': True, 'n_jobs': 1, 'positive': False} - best score: 0.14821434561501406
```
- le score est ridicule, bon on continu quand même

### Train model
- On fait tourner le modéle avec les données X_train_scaled.csv

### Evaluate Model
- On evalue le modèle avec X_test_scaled.scv
- On les compare avec y_test

### Metric
- Le model LinearRegression n'est pas bon du tout !!!
```
{
    "explained_variance": 0.1507,
    "mean_squared_log_error": 0.0823,
    "r2": 0.1506,
    "MAE": 0.7585,
    "MSE": 0.96,
    "RMSE": 0.9798
}

```
### Mise en place du Pipline dvc.yaml
- 3 etapes : prepare - train - evaluate
- remove all artifact dvc output
```
$  dvc remove data/processed models/lr_model.joblib.dvc models/lr_grid_search_estimator.pkl.dvc models/lr_randon_search_estimator.pkl.dvc data/processed.dvc

```
- stage prepare
```
dvc stage add -n prepare \                             
-d src/data/make_dataset.py -d src/data/normalized_dataset.py -d data/raw \
-o data/processed \
python src/data/make_dataset.py \
python src/data/normalized_dataset.py \
dvc repro
```
- stage find_best_hyperparametre
```
dvc stage add -n hyperparameter \
-d src/models/find_linear_regression_parameters.py \
-d data/processed \
-o models/lr_grid_search_estimator.pkl \v 
-o models/lr_grid_randonh_estimator.pkl \
python src/models/find_linear_regression_parameters.py
```
- pour les stages train et evalutate voir dvc.yml
```
stages:
  prepare:
    cmd:
    - python src/data/make_dataset.py
    - python src/data/normalized_dataset.py
    deps:
    - data/raw
    - src/data/make_dataset.py
    - src/data/normalized_dataset.py
    outs:
    - data/processed
  hyperparameter:
    cmd: python src/models/find_linear_regression_parameters.py
    deps:
    - data/processed
    - src/models/find_linear_regression_parameters.py
    outs:
    - models/lr_grid_search_estimator.pkl
    - models/lr_random_search_estimator.pkl
  train:
    cmd:
    - python src/models/train_linear_regression_model.py
    deps:
    - data/processed
    - src/models/train_linear_regression_model.py
    outs:
    - models/lr_model.joblib
  evaluate:
    cmd:
    - python src/models/evaluate_linear_regression_model.py
    deps:
    - data/processed
    - src/models/evaluate_linear_regression_model.py
    outs:
    - metrics/lr_metrics.json
    
```