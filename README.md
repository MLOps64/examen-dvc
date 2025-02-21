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
    - preprocessed
### Installation DVC
```
pip3 instal dvc
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
```
