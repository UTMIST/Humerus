import pathlib

from transformers import (DataCollatorForLanguageModeling, GPT2LMHeadModel,
                          GPT2Tokenizer, LineByLineTextDataset, Trainer,
                          TrainingArguments)


def finetune_gpt2(data_path, save_dir):

    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=str(data_path),
        block_size=128,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # TODO: set configurations here
    training_args = TrainingArguments(
        output_dir=str(save_dir),
        overwrite_output_dir=True,
        num_train_epochs=100,
        per_gpu_train_batch_size=64,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        prediction_loss_only=True,
    )

    trainer.train()

    trainer.save_model(save_dir)


if __name__ == '__main__':

    PROJ_DIR = pathlib.Path(__file__).parents[1]
    DATA_DIR = PROJ_DIR / 'datasets' / 'processed' / 'train.txt'
    SAVE_DIR = PROJ_DIR / 'models'

    finetune_gpt2(DATA_DIR, SAVE_DIR)
