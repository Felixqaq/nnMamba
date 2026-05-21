# Config: regression/environment.tapct.yml

- Source: [regression/environment.tapct.yml](/home/felix/Research/nnMamba/regression/environment.tapct.yml)
- Size: 496 B

## Parsed Summary
| field | value |
| --- | --- |
| task |  |
| model.name |  |
| data.target_mode |  |
| training.epochs |  |
| training.batch_size |  |
| training.k_folds |  |
| training.learning_rate |  |
| training.loss |  |
| data.balanced_sampling |  |
| data.augmentation.enabled |  |
| experiment.name |  |

## Full YAML
```yaml
name: tapct
channels:
  - defaults
dependencies:
  - python=3.11
  - pip
  - pip:
      - --index-url
      - https://download.pytorch.org/whl/cu128
      - torch==2.8.0
      - torchvision==0.23.0
      - --extra-index-url
      - https://pypi.org/simple
      - transformers>=4.50
      - huggingface_hub
      - safetensors
      - SimpleITK>=2.5
      - monai>=1.4
      - scikit-learn
      - pandas
      - matplotlib
      - tqdm
      - pyyaml
      - nibabel
      - timm
      - einops

```
