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

Возможно, нам нужно залезть в директорию trainers, но сначала надо подумать про конфиг и как вообще тут решение собирается. Мб есть смысл воспроизвести их SFT как шаг перед alignment. Как я поняла, мы забили на SFT, то есть в бейзлайне только претрейн и мы сразу делаем алаймент  

## Customized datasets

For supported datasets, refer to [`REAL_DATASET_CONFIGS(src/data/configs.py)`](https://github.com/ZHZisZZ/modpo/blob/main/src/data/configs.py#L19).
To train on your datasets, add them under [`src/data/raw_data`](https://github.com/ZHZisZZ/modpo/blob/main/src/data/raw_data) and modify [`REAL_DATASET_CONFIGS(src/data/configs.py)`](https://github.com/ZHZisZZ/modpo/blob/main/src/data/configs.py#L19) accordingly. Please see [`src/data/raw_data/shp`](https://github.com/ZHZisZZ/modpo/blob/main/src/data/raw_data/shp.py) for an example.

У нас свой датасет. Скорее всего, нужно сделать как они предлагают 

# Примерное решение от ChatGPT
_Нужно иметь в виду, что нам нужно сделать MODPO по своим метрикам, а в статье делают алаймент для целей AI safety_

Исходя из загруженной вами статьи по **MODPO** (Multi-Objective Direct Preference Optimization), основная идея заключается в том, чтобы выравнивать модели по нескольким предпочтениям без использования RLHF (Reinforcement Learning from Human Feedback). В вашей задаче вы хотите применить этот подход к **машинному переводу**, чтобы учесть различные предпочтения в переводах (например, точность, стиль, краткость).

Вот несколько ключевых шагов, как можно адаптировать подход **MODPO** для вашей задачи машинного перевода:

### 1. **Подготовка данных**
Как и в примерах из статьи, вам нужно собрать отзывы по разным метрикам качества перевода. Например:
- **Точность перевода**: насколько перевод соответствует исходному смыслу.
- **Краткость/конкретность**: возможно, необходимо избегать избыточных пояснений.
- **Естественность**: насколько результат перевода звучит естественно на целевом языке.

Для каждой из этих метрик вам нужно собрать данные, которые будут использоваться как предпочтения для обучения.

### 2. **Использование модели с Hugging Face**
У вас уже есть обученная модель, что упрощает задачу. Чтобы применить **MODPO** для настройки её поведения, следуйте этим шагам:

- **Обучение модели вознаграждений**:
  - На основе собранных предпочтений обучите несколько моделей вознаграждений, каждая из которых будет оценивать разные аспекты перевода. Например, одна модель может оценивать точность, другая — краткость, и т.д.
  
- **Применение MODPO**:
  - Используйте подход MODPO для дальнейшего обучения вашей модели машинного перевода, как указано в статье:
  
    ```bash
    PYTHONPATH=. accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=8 \
        scripts/modpo/beavertails/modpo.py \
        --sft_model_name "имя_вашей_модели_с_Hugging_Face" \
        --margin_reward_model_name "./путь_к_модели_вознаграждений" \
        --prompt_template "BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:" \
        --dataset_name "имя_вашего_датасета" \
        --max_length 512 \
        --w 0.5 \
        --training_args.output_dir "./output/modpo_translation" \
        --training_args.run_name "modpo_translation_run" \
        --training_args.per_device_train_batch_size 6 \
        --training_args.per_device_eval_batch_size 6 \
        --training_args.gradient_accumulation_steps 2 \
        --training_args.learning_rate 5e-4 \
        --peft_config.r 64 \
        --peft_config.target_modules q_proj k_proj v_proj o_proj \
        --peft_config.lora_alpha 1 \
        --peft_config.lora_dropout 0
    ```

- **Балансировка предпочтений**: Используя веса (`w`) для разных моделей вознаграждений, вы можете настраивать, насколько сильно учитываются те или иные аспекты (точность, краткость, естественность и т.д.).

### 3. **Оценка результатов**
- После обучения, используйте генерацию результатов и их оценку, чтобы проверить, насколько хорошо модель соблюдает баланс между точностью, краткостью и естественностью. Вы можете применить подобные скрипты:

    ```bash
    PYTHONPATH=. python3 scripts/modpo/beavertails/utils/gen.py \
        --sft_model_name "имя_вашей_модели_с_Hugging_Face" \
        --adapter_model_name "./output/modpo_translation/best_checkpoint" \
        --prompt_template "BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:" \
        --dataset_name "имя_вашего_датасета" \
        --output_dir "./output/modpo_translation/gen" \
        --max_length 512
    ```

### 4. **Дополнительные эксперименты и настройка**
- Экспериментируйте с различными комбинациями предпочтений и весов для каждой задачи машинного перевода. Используйте кросс-валидацию и обратную связь от реальных пользователей, если это возможно.

Этот процесс позволит вам настроить модель машинного перевода так, чтобы она соответствовала разным целям выравнивания (например, стилистика, краткость, точность), используя подходы MODPO.

# Я хочу сначала воспроизвести решение авторов MODPO, чтобы убедиться, что ничего не падает, и разобраться, как тут всё работает. Потом заняться переводом (модифицировать датасеты и т п)

## Как сохранить веса и потом использовать снова, чтобы не терять компьют?
Для того чтобы не тратить вычислительные ресурсы на повторное обучение модели, вы можете сохранить результаты обучения и затем загрузить их для дальнейшего использования. Вот основные шаги, как это сделать:

### 1. **Сохранение модели после обучения**

В процессе обучения вашей модели с использованием `DPO`, результаты обучения (веса модели и другие артефакты) будут сохраняться в указанный вами выходной каталог. Убедитесь, что у вас настроено сохранение модели:

- Ваша команда для обучения, как уже указано, сохраняет результаты в:
  ```bash
  --training_args.output_dir "./output/PKU-Alignment/PKU-SafeRLHF-10K/modpo/rm/safer"
  ```

Убедитесь, что в этом каталоге находятся файлы модели (например, `pytorch_model.bin`, `config.json` и т.д.).

### 2. **Загрузка модели для дальнейшего использования**

Когда вам нужно будет использовать уже обученную модель, вы можете загрузить её в своем коде. Например, если вы используете библиотеку `transformers` от Hugging Face, вы можете загрузить модель следующим образом:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Укажите путь к вашей сохраненной модели
model_path = "./output/PKU-Alignment/PKU-SafeRLHF-10K/modpo/rm/safer/best_checkpoint"

# Загрузка модели и токенизатора
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
```

### 3. **Использование модели для генерации ответов**

Теперь, когда ваша модель загружена, вы можете использовать её для генерации ответов без необходимости повторного обучения:

```python
# Пример использования модели для генерации текста
prompt = "BEGINNING OF CONVERSATION: USER: Привет, как дела? ASSISTANT:"
inputs = tokenizer(prompt, return_tensors="pt")

# Генерация ответа
outputs = model.generate(**inputs, max_length=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
```

### 4. **Экспорт и импорт моделей**

Если вы хотите сохранить модели для использования на других системах или в облаке:

- **Экспорт**: Скопируйте файлы модели из выходного каталога на ваш локальный диск или в облачное хранилище.
- **Импорт**: Загружайте их так, как показано выше.

### 5. **Сохранение и загрузка через Hugging Face Hub (опционально)**

Если вы используете Hugging Face Hub, вы можете также загрузить вашу модель на этот репозиторий:

```bash
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./output/PKU-Alignment/PKU-SafeRLHF-10K/modpo/rm/safer",
    repo_id="your_username/your_model_name",
    commit_message="Upload trained model"
)
```

А затем вы сможете загружать её откуда угодно, используя:

```python
model = AutoModelForCausalLM.from_pretrained("your_username/your_model_name")
tokenizer = AutoTokenizer.from_pretrained("your_username/your_model_name")
```

### Заключение

Сохранение и загрузка модели - это отличный способ оптимизировать ваше использование вычислительных ресурсов. Это позволяет вам повторно использовать уже обученные модели без необходимости проходить через весь процесс обучения каждый раз.
