# AGENTS.md

## Titanic Classification

### Data Files
- `train_data.csv` — training data (792 rows)
- `test_data.csv` — test data (100 rows, **includes `Survived` labels**)

### Data Schema
- **Label:** `Survived` (binary: 0/1)
- **Features (14):** `Sex`, `Age`, `Fare`, `Pclass_1/2/3`, `Family_size`, `Title_1-4`, `Emb_1-3`
- **Preprocessed:** numeric features scaled (0-1 range), categoricals one-hot encoded
- **No missing values** in current files
- First unnamed column is an index; exclude from features

### Workflow
1. Load `train_data.csv` and `test_data.csv`
2. Train RandomForestClassifier (`random_state=42`)
3. Evaluate on test data (confusion matrix, AUC score)
4. Export `predictions.csv` with original test data + `predicted_label` column

