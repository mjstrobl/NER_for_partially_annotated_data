import logging
import os
import torch
import numpy as np
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from metrics_vanilla import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)

def train(results_dir, original_output_dir, max_steps, save_steps, evaluate_during_training, datasets, model, tokenizer, pad_token_label_id, label_list, per_gpu_train_batch_size, gradient_accumulation_steps, num_train_epochs, weight_decay, learning_rate, adam_epsilon, warmup_steps, model_name_or_path, device, model_type, max_grad_norm, logging_steps):
    train_dataset = datasets['train']
    train_batch_size = per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(model_name_or_path, "scheduler.pt")))

    # Train!
    print("***** Running training *****")
    print("  Num examples = " + str(len(train_dataset)))
    print("  Num Epochs = " +  str(num_train_epochs))
    print("  Instantaneous batch size per GPU = " + str(per_gpu_train_batch_size))
    print(
        "  Total train batch size (w. parallel, distributed & accumulation) = " +
        str(train_batch_size
        * gradient_accumulation_steps))
    print("  Gradient Accumulation steps = " +  str(gradient_accumulation_steps))
    print("  Total optimization steps = " +  str(t_total))

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // gradient_accumulation_steps)

        print("  Continuing training from checkpoint, will skip to saved global_step")
        print("  Continuing training from epoch " +  str(epochs_trained))
        print("  Continuing training from global step " +  str(global_step))
        print("  Will skip the first " + str(steps_trained_in_current_epoch) + " steps in the first epoch")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(num_train_epochs), desc="Epoch")

    for _ in train_iterator:
        current_loss = 0.0

        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            current_loss += loss.item()
            epoch_iterator.set_description('loss=%g' % (current_loss / (step + 1)))

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if save_steps > 0 and global_step % save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(original_output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    print("Saving model checkpoint to " + output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    print("Saving optimizer and scheduler states to " + output_dir)

            if max_steps > 0 and global_step > max_steps:
                epoch_iterator.close()
                break

        if (evaluate_during_training):  # Only evaluate when single GPU otherwise metrics may not average well
            evaluate(results_dir, datasets, model, per_gpu_train_batch_size, device, model_type, pad_token_label_id, label_list,prefix='dev')
            evaluate(results_dir, datasets, model, per_gpu_train_batch_size, device, model_type, pad_token_label_id, label_list,prefix='gold')

        if max_steps > 0 and global_step > max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def evaluate(results_directory, datasets, model, per_gpu_eval_batch_size, device, model_type, pad_token_label_id, label_list,prefix="dev"):
    eval_dataset = datasets[prefix]
    all_tokens = datasets[prefix + "_tokens"]

    #sorted(features, key = lambda feature : feature.length, reverse=True)

    eval_batch_size = per_gpu_eval_batch_size
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    # Eval!
    print("***** Running evaluation on " + prefix + " dataset *****")
    print("  Num examples = " + str(len(eval_dataset)))
    print("  Batch size = " + str(eval_batch_size))
    #eval_loss = 0.0
    nb_eval_steps = 0
    raw_preds = None
    num_labels = 11
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            if model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids

            outputs = model(**inputs)
            logits = outputs[0]

            #eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if raw_preds is None:
            raw_preds = logits.detach().cpu().numpy()
            out_label_ids = batch[3].detach().cpu().numpy()
        else:
            raw_preds = np.append(raw_preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, batch[3].detach().cpu().numpy(), axis=0)

    #eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(raw_preds, axis=2)

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    label_map = {i: label for i, label in enumerate(label_list)}

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])
            else:
                out_label_list[i].append(label_map[0])
                preds_list[i].append(label_map[0])

    with open(results_directory + prefix + '_vanilla.txt','w') as f:
        for i in range(out_label_ids.shape[0]):
            tokens = all_tokens[i]
            original_labels = []
            predicted_labels = []
            correct = True
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] >= 0:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])
                    if out_label_list[i][-1] != preds_list[i][-1]:
                        correct = False
                    original_labels.append(out_label_list[i][-1])
                else:
                    original_labels.append('-')

                if j < len(tokens):
                    if tokens[j].startswith('#'):
                        predicted_labels.append('-')
                    else:
                        predicted_labels.append(label_map[preds[i][j]])



            if not correct:
                for j in range(len(tokens)):
                    f.write(tokens[j] + ' (' + original_labels[j] + '|' + predicted_labels[j] + ') ')
                f.write('\n')



    types = ['PER', 'LOC', 'ORG', 'MISC','DRUG']
    for type in types:
        print("type: " + type)
        results = {
            #"loss": eval_loss,
            "precision": precision_score(out_label_list, preds_list, type=type),
            "recall": recall_score(out_label_list, preds_list, type=type),
            "f1": f1_score(out_label_list, preds_list, type=type),
        }

        print("***** " + type + " results " + prefix + " *****")
        for key in sorted(results.keys()):
            print("  " + key + " = " + str(results[key]))

    results = {
        #"loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }

    print("***** Eval results " + prefix + " *****")
    for key in sorted(results.keys()):
        print("  " + key + " = " + str(results[key]))

    return results
