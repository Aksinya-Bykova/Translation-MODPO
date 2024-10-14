# Running MODPO
**ChatGPT messages, need to check**

[scripts/modpo/beavertails]()

Этот пример направлен на выравнивание модели по критериям безопасности.
В частности, задача состоит в балансировке различных ценностей, таких как безопасность и полезность. Это может быть актуально для моделей, которым нужно соблюдать правила безопасности (например, избегать вредных рекомендаций), оставаясь при этом полезными для пользователя.

[scripts/modpo/summarize_w_length_penalty]()

Эта задача связана с устранением предвзятости к длине в задачах суммаризации текста.
Модель должна избегать излишней многословности, выдавая более краткие и точные резюме. Для этого применяется штраф за длину (length penalty), который помогает уменьшить склонность модели к излишнему развертыванию ответов.

## Other examples

This repository also contains other off-the-shelf tuning recipes:

- SFT (Supervised Fine-tuning): [`scripts/examples/sft/run.sh`](https://github.com/ZHZisZZ/modpo/blob/main/scripts/examples/sft/run.sh)
- RM (Reward Modeling): [`scripts/examples/rm/run.sh`](https://github.com/ZHZisZZ/modpo/blob/main/scripts/examples/rm/run.sh)
- DPO (Direct Preference Optimization): [`scripts/examples/dpo/run.sh`](https://github.com/ZHZisZZ/modpo/blob/main/scripts/examples/dpo/run.sh)

To implement new alignment algorithms, please add new trainers at [`src/trainer`](https://github.com/ZHZisZZ/modpo/blob/main/src/trainer).

Возможно, нам нужно залезть в эту директорию. 

## Customized datasets

For supported datasets, refer to [`REAL_DATASET_CONFIGS(src/data/configs.py)`](https://github.com/ZHZisZZ/modpo/blob/main/src/data/configs.py#L19).
To train on your datasets, add them under [`src/data/raw_data`](https://github.com/ZHZisZZ/modpo/blob/main/src/data/raw_data) and modify [`REAL_DATASET_CONFIGS(src/data/configs.py)`](https://github.com/ZHZisZZ/modpo/blob/main/src/data/configs.py#L19) accordingly. Please see [`src/data/raw_data/shp`](https://github.com/ZHZisZZ/modpo/blob/main/src/data/raw_data/shp.py) for an example.

У нас свой датасет. Скорее всего, нужно сделать как они предлагают 
