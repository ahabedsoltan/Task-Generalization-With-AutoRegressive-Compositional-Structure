from transformers import Trainer
import transformers
import numpy as np
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils import set_seed, parse_args
from data import load_dataset
from model import get_model

def main():
    args = parse_args()
    set_seed(args.seed)
    train_dataset = load_dataset(args.dataset_dir, args.dataset_type)
    val_dataset = load_dataset(os.path.join(args.dataset_dir, 'val'), args.dataset_type)
    model_args = eval(open(args.model_config_path).read())
    model_args["num_hidden_layers"] = args.num_hidden_layers
    model_args["num_attention_heads"] = args.num_attention_heads
    print(model_args)
    model = get_model(
        **model_args
    )
    if(args.model_dir):
        import safetensors
        safetensors.torch.load_model(model, os.path.join(args.model_dir, 'model.safetensors'))
    output_dir = f"{args.output_dir}{args.dataset_dir.split('/')[-1]}_{args.total_training_samples}_LR={args.lr}_WD={args.weight_decay}_{args.world_size}GPU*{args.batch_size}Batch_{args.model_config_path.split('/')[-1]}_#layer={args.num_hidden_layers}_#head={args.num_attention_heads}"
    training_args = transformers.TrainingArguments(
        output_dir=output_dir,          
        num_train_epochs=args.total_training_samples / len(train_dataset),              
        per_device_train_batch_size=args.batch_size,  
        per_device_eval_batch_size=args.batch_size,   
        warmup_steps=0,                
        weight_decay=args.weight_decay,               
        logging_dir='./logs',            
        logging_steps= args.log_interval // (args.batch_size * args.world_size),
        save_steps = args.save_interval // (args.batch_size *args.world_size),
        save_total_limit = 1,
        evaluation_strategy="steps",     
        eval_steps= args.eval_interval // (args.batch_size * args.world_size), 
        learning_rate = args.lr,
        label_names = ['labels'],
        save_safetensors = False,
        report_to = "none",
    )
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if model_args["model_type"] == 'gpt2_custom_simpler':
            predictions = np.squeeze((predictions >= 0.5).astype(int))
            print(predictions)
        else:
            predictions = np.argmax(predictions, axis=-1)
        predictions = predictions[:, :-1]
        labels = labels[:, 1:]
        exact_match_cnt = 0
        cnt = 0
        for prediction, label in zip(predictions, labels):
            correct = (prediction == label) + (label == -100)
            cnt += 1
            exact_match_cnt += correct.all()
        return {"exact_match": exact_match_cnt / cnt}
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        eval_dataset = val_dataset,
    )
    trainer.train(ignore_keys_for_eval = ['past_key_values', 'dreamer_loss_1', 'dreamer_loss_0'])
    trainer.save_model(output_dir=args.output_dir)

if __name__ == '__main__':
    main()