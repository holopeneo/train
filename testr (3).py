import os
import time
from typing import Dict, Optional

import evaluate
import numpy as np
import pandas as pd
import torch
from aim import Run
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


# ======================
# Конфигурация
# ======================
MODEL_NAME = "LGAI-EXAONE/EXAONE-4.0-1.2B"
CSV_PATH = "ads_summarized_train_partial_9000.csv"
OUTPUT_DIR = "./exaone-ads-snippets-qlora-fp16"

MAX_LENGTH = 512
TEST_SIZE = 0.1
SEED = 42

NUM_EPOCHS = 10
BATCH_SIZE = 4
GRAD_ACCUM = 4
LR = 2e-4

# Системный промпт (одинаковый для обучения и генерации)
SYSTEM_PROMPT = (
    "Ты — профессиональный редактор, создающий краткие аннотации для объявлений.\n"
    "Создай саммари из тайтла и описания объявления, соблюдая правила:\n"
    "- Саммари должно быть 1–2 предложениями (максимум 200 слов)\n"
    "- Сохрани ключевую информацию: что продается/предлагается, главные характеристики, преимущества\n"
    "- Используй ясный и лаконичный язык\n"
    "- НЕ добавляй информации, которой нет в исходном тексте\n"
    "- Избегай оценочных суждений вроде 'топовый', 'лучший'. Только факты.\n"
)

# Aim Run
run = Run()
run["hparams"] = {
    "model": MODEL_NAME,
    "method": "QLoRA",
    "max_length": MAX_LENGTH,
    "num_epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "grad_accum": GRAD_ACCUM,
    "learning_rate": LR,
}


# ======================
# Aim Callback
# ======================
class AimLoggingCallback(TrainerCallback):
    """Пишет числовые логи Trainer в Aim, использует epoch как шаг."""

    def __init__(self):
        self.epoch_start_time: Optional[float] = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        step = logs.get("epoch", state.epoch)
        if step is None:
            step = state.global_step
        step = int(step or 0)

        for key, value in logs.items():
            if isinstance(value, (int, float)):
                run.track(value, name=key, step=step)

    def on_epoch_end(self, args, state, control, **kwargs):
        step = int(state.epoch or state.global_step or 0)
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0.0
        run.track(epoch_time, name="epoch_time_seconds", step=step)


# ======================
# Генерация и метрики ROUGE/BERTScore
# ======================
class GenerationMetrics:
    """Считает метрики на сыром eval датасете через генерацию ответов модели."""

    def __init__(
        self,
        raw_eval_dataset,
        tokenizer,
        system_prompt: str,
        max_eval_samples: Optional[int] = None,
        max_new_tokens: int = 200,
        batch_size: int = 8,
    ):
        self.raw_eval_dataset = raw_eval_dataset
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.max_eval_samples = max_eval_samples
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

        self.rouge = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")

    def compute(self, model) -> Dict[str, float]:
        model.eval()
        dataset = self.raw_eval_dataset
        if self.max_eval_samples is not None and self.max_eval_samples < len(dataset):
            dataset = dataset.select(range(self.max_eval_samples))

        references = []
        predictions = []

        for i in tqdm(range(0, len(dataset), self.batch_size), desc="[GenerationMetrics] Generating summaries"):
            batch = dataset[i : i + self.batch_size]
            subjects = batch["subject"]
            infos = batch["info"]
            refs = batch["summary"]
            references.extend(refs)

            prompts = []
            for subj, info in zip(subjects, infos):
                user_prompt = f"Тайтл: {subj}\nОписание: {info}\n\nСаммари:\n    \"\"\"\n"
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompts.append(prompt)

            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            for j in range(len(prompts)):
                prompt_len = inputs["attention_mask"][j].sum().item()
                gen_ids = outputs[j, prompt_len:]
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                predictions.append(text.strip())

        rouge_results = self.rouge.compute(predictions=predictions, references=references)
        bert_results = self.bertscore.compute(predictions=predictions, references=references, lang="ru")

        return {
            "eval_rouge1": float(rouge_results["rouge1"]),
            "eval_rouge2": float(rouge_results["rouge2"]),
            "eval_rougeL": float(rouge_results["rougeL"]),
            "eval_rougeLsum": float(rouge_results["rougeLsum"]),
            "eval_bert_precision": float(np.mean(bert_results["precision"])),
            "eval_bert_recall": float(np.mean(bert_results["recall"])),
            "eval_bert_f1": float(np.mean(bert_results["f1"])),
        }


# ======================
# Данные
# ======================
def load_and_filter_data(csv_path: str) -> DatasetDict:
    print(f"Загружаю CSV из {csv_path}...")
    df = pd.read_csv(csv_path)

    df = df[["subject", "info", "summary"]]
    df = df[df["summary"].notna()]
    df["summary"] = df["summary"].astype(str).str.strip()
    df = df[df["summary"] != ""]

    df["subject"] = df["subject"].fillna("")
    df["info"] = df["info"].fillna("")

    print(f"После фильтрации осталось {len(df)} объявлений")

    ds = Dataset.from_pandas(df, preserve_index=False)
    ds = ds.train_test_split(test_size=TEST_SIZE, seed=SEED)

    return DatasetDict(train=ds["train"], test=ds["test"])


def make_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def tokenize_dataset(examples, tokenizer):
    texts = []
    for subject, info, summary in zip(examples["subject"], examples["info"], examples["summary"]):
        user_prompt = f"Тайтл: {subject}\nОписание: {info}\n\nСаммари:\n    \"\"\"\n"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": summary},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)

    tokenized = tokenizer(
        texts,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


# ======================
# QLoRA модель
# ======================
def load_qlora_model_fp16(model_name: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("Загружаю EXAONE с QLoRA FP16…")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


# ======================
# Кастомный Trainer
# ======================
class CustomTrainer(Trainer):
    def __init__(self, *args, generation_metrics: Optional[GenerationMetrics] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.generation_metrics = generation_metrics

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)

        if self.generation_metrics is not None:
            gen_metrics = self.generation_metrics.compute(self.model)
            metrics.update(gen_metrics)
            self.log(gen_metrics)

        return metrics


# ======================
# main
# ======================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = make_tokenizer(MODEL_NAME)
    raw_datasets = load_and_filter_data(CSV_PATH)

    tokenize_fn = lambda examples: tokenize_dataset(examples, tokenizer)
    tokenized_train = raw_datasets["train"].map(
        tokenize_fn,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Токенизация train",
    )
    tokenized_test = raw_datasets["test"].map(
        tokenize_fn,
        batched=True,
        remove_columns=raw_datasets["test"].column_names,
        desc="Токенизация test",
    )
    tokenized_datasets = DatasetDict(train=tokenized_train, test=tokenized_test)

    model = load_qlora_model_fp16(MODEL_NAME)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args_kwargs = dict(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=LR,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        fp16=True,
        bf16=False,
        save_total_limit=3,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_bert_f1",
        greater_is_better=True,
    )

    try:
        # Новые версии transformers используют evaluation_strategy
        training_args = TrainingArguments(
            evaluation_strategy="epoch",
            **training_args_kwargs,
        )
    except TypeError:
        # Старые версии могут ожидать eval_strategy
        training_args = TrainingArguments(
            eval_strategy="epoch",
            **training_args_kwargs,
        )

    aim_callback = AimLoggingCallback()
    generation_metrics = GenerationMetrics(
        raw_eval_dataset=raw_datasets["test"],
        tokenizer=tokenizer,
        system_prompt=SYSTEM_PROMPT,
        max_eval_samples=None,
        max_new_tokens=200,
        batch_size=8,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        callbacks=[aim_callback],
        generation_metrics=generation_metrics,
    )

    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nОбучение завершено! Модель сохранена в:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
