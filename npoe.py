import argparse
import torch
from PackDataset import packDataset_util_bert
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForSequenceClassification
from transformers import (
   AdamW, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
   )
import os
from torch.nn.utils import clip_grad_norm_
import csv
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import random
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import statistics 
import scipy.stats as st
import time


def read_data(file_path):
   import pandas as pd
   data = pd.read_csv(file_path, sep='\t').values.tolist()
   processed_data=[]
   for item in data:
      if len(item)<2:
         break
      processed_data.append((item[0], int(item[1])))
   return processed_data


def get_all_data(base_path):
   train_path = os.path.join(base_path, 'train.tsv')
   dev_path = os.path.join(base_path, 'dev.tsv')
   test_path = os.path.join(base_path, 'test.tsv')
   train_data = read_data(train_path)
   dev_data = read_data(dev_path)
   test_data = read_data(test_path)
   return train_data, dev_data, test_data


def evaluation(loader, test_model):
   test_model.eval()
   total_number = 0
   total_correct = 0
   logits=[]
   with torch.no_grad():
      for padded_text, attention_masks, labels in loader:
         if torch.cuda.is_available():
            padded_text,attention_masks, labels = padded_text.cuda(), attention_masks.cuda(), labels.cuda()
         output = test_model(padded_text, attention_masks)[0]
         _, idx = torch.max(output, dim=1)
         correct = (idx == labels).sum().item()
         total_correct += correct
         total_number += labels.size(0)
         
         probs = F.softmax(output, dim=1)
         for i,l in enumerate(labels):
            logits.append(probs[i][l].item())
      acc = total_correct / total_number
   return acc, logits
      
def evaluation_whole_bias(loader, bias_models):
   for b in bias_models:
      b.eval()
   total_number = 0
   total_correct = 0
   logits=[]
   with torch.no_grad():
      for padded_text, attention_masks, labels in loader:
         if torch.cuda.is_available():
            padded_text,attention_masks, labels = padded_text.cuda(), attention_masks.cuda(), labels.cuda()
         
         bias_outputs = torch.empty((args.num_bias_experts, padded_text.size()[0], num_classes), dtype=torch.float64)
         if torch.cuda.is_available():
            bias_outputs = bias_outputs.cuda()
         for i, bias_model in enumerate(bias_models):
            bias_outputs[i] = bias_model(padded_text, attention_masks)[0]
            
         # other gate settings ommitted for clarity
         gate_output = gate_model(padded_text, attention_masks)[0]         
         _, idx = gate_output.topk(args.top_k_routing) 
         mask = torch.zeros_like(gate_output)
         for i,j in enumerate(idx):
            mask[i][j] = 1
         gate_output *= mask
         gate_output = F.softmax(gate_output, dim=1)
         
         bias_term = None
         for i, out in enumerate(bias_outputs):
            l = F.softmax(out, dim=1)
            if bias_term==None:
               bias_term = l*gate_output[:,i,None]
            else:
               bias_term += l*gate_output[:,i,None]
         
         _, idx = torch.max(bias_term, dim=1)
         correct = (idx == labels).sum().item()
         total_correct += correct
         total_number += labels.size(0)
         
         for i,l in enumerate(labels):
            logits.append(bias_term[i][l].item())
      acc = total_correct / total_number
      return acc, logits
      
def evaluation_pseudo(loader):
    model.eval()
    for i, bias_model in enumerate(bias_models):
      bias_model.eval()
    total_correct_asr = 0
    total_selected_asr = 0
    total_correct_acc = 0
    total_selected_acc = 0
    with torch.no_grad():
        for padded_text, attention_masks, labels in loader:
            if torch.cuda.is_available():
                padded_text, attention_masks, labels = padded_text.cuda(), attention_masks.cuda(), labels.cuda()
                
            output_main = model(padded_text, attention_masks)[0]
            
            bias_outputs = torch.empty((args.num_bias_experts, padded_text.size()[0], num_classes), dtype=torch.float64)
            if torch.cuda.is_available():
               bias_outputs = bias_outputs.cuda()
            for i, bias_model in enumerate(bias_models):
               bias_outputs[i] = bias_model(padded_text, attention_masks)[0]
               
            # other gate settings ommitted for clarity
            gate_output = gate_model(padded_text, attention_masks)[0]         
            _, idx = gate_output.topk(args.top_k_routing) 
            mask = torch.zeros_like(gate_output)
            for i,j in enumerate(idx):
               mask[i][j] = 1
            gate_output *= mask
            gate_output = F.softmax(gate_output, dim=1)
            
            bias_term = None
            for i, out in enumerate(bias_outputs):
               l = F.softmax(out, dim=1)
               if bias_term==None:
                  bias_term = l*gate_output[:,i,None]
               else:
                  bias_term += l*gate_output[:,i,None]
            output_bias = bias_term
            
                         
            t, idx_m = torch.max(output_main, dim=1)
            _, idx_b = torch.max(output_bias, dim=1) 
            
            labels2 = torch.reshape(labels, (labels.size()[0],1))
            confidence_m = torch.gather(output_main, 1, labels2)
            confidence_b = torch.gather(output_bias, 1, labels2)
            confidence_m = torch.squeeze(confidence_m)
            confidence_b = torch.squeeze(confidence_b)
            
            confidence_m = confidence_m.cpu().numpy()
            confidence_b = confidence_b.cpu().numpy()
            idx_m = idx_m.cpu().numpy()
            idx_b = idx_b.cpu().numpy()
            labels = labels.cpu().numpy()
            correct = np.sum((idx_m == labels) & (confidence_b >= args.pseudo_dev_thre_high) & (confidence_m < args.thre_main_high))
            total_correct_asr += correct
            correct = np.sum((idx_m == labels) & (confidence_b < args.pseudo_dev_thre_low) & (confidence_m > args.thre_main_low))
            total_correct_acc += correct
            selected = np.sum((confidence_b >= args.pseudo_dev_thre_high) & (confidence_m < args.thre_main_high))
            total_selected_asr += selected
            selected = np.sum((confidence_b < args.pseudo_dev_thre_low) & (confidence_m > args.thre_main_low))
            total_selected_acc += selected
        print('total_correct_asr, total_selected_asr, total_correct_acc, total_selected_acc:', total_correct_asr, total_selected_asr, total_correct_acc, total_selected_acc) 
        write_results(['(total_correct_asr, total_selected_asr, total_correct_acc, total_selected_acc)', total_correct_asr, total_selected_asr, total_correct_acc, total_selected_acc]) 

    return total_correct_asr, total_selected_asr, total_correct_acc, total_selected_acc


def poe_loss(output, bias_outputs, gate_output, labels):
   """Implements the product of expert loss."""
   pt = F.softmax(output, dim=1)
   
   if not args.no_gate:
      bias_term = None
      for i, out in enumerate(bias_outputs):
         pt_2 = F.softmax(out, dim=1)
         l = torch.log(pt_2)
         if bias_term==None:
            bias_term = l*gate_output[:,i,None]
         else:
            bias_term += l*gate_output[:,i,None]
   else:
      r=random.randint(0, args.num_bias_experts-1)
      out = bias_outputs[r]
      pt_2 = F.softmax(out, dim=1)
      bias_term = torch.log(pt_2)
   joint_pt = F.softmax((torch.log(pt) + poe_alpha * bias_term), dim=1)
   joint_p = joint_pt.gather(1, labels.view(-1, 1))
   batch_loss = -torch.log(joint_p)
   if args.kl_div:
      label_div = F.kl_div(bias_outputs[0], bias_outputs[1])
      batch_loss -= 0.000001 * label_div
   loss = batch_loss.mean()
   return loss
   
   
def plain_loss(output, labels):
   """No PoE loss."""
   pt = F.softmax(output, dim=1)
   joint_p = pt.gather(1, labels.view(-1, 1))
   batch_loss = -torch.log(joint_p)
   loss = batch_loss.mean()
   return loss
      


def kl_loss(p, q):
   p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
   q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

   # You can choose whether to use function "sum" and "mean" depending on your task
   p_loss = p_loss.mean()
   q_loss = q_loss.mean()

   loss = (p_loss + q_loss) / 2
   return loss

def pretrain_loss(bias_output, labels):
   CE_loss = nn.BCELoss()
   return CE_loss(bias_output, labels)  
   

def pretrain():
   train_loaders = [] 
   if 'sentence' not in args.ablate_trigger:
      train_loaders.append((sent_train_loader_poison_pretrain, 'sentence'))
   if 'syntax' not in args.ablate_trigger:
      train_loaders.append((syn_train_loader_poison_pretrain, 'syntax'))
   if 'word' not in args.ablate_trigger:
      train_loaders.append((word_train_loader_poison_pretrain, 'word'))
   if args.num_bias_experts>3 and args.style_poison_data_path!=None and 'style' not in args.ablate_trigger:
      train_loaders.append((style_train_loader_poison_pretrain, 'style'))
   try:
      print('start pretraining bias models')
      write_results(['start pretraining'])
      flag = 0
      for b, bias_model in enumerate(bias_models):
         print('pretraining bias model', b)
         total_loss = 0
         CE_loss = nn.CrossEntropyLoss()
         for epoch in tqdm(range(EPOCHS_PRETRAIN)):
            iter = 0
            bias_model.train()
            NUM_EX = len(train_loaders[b%len(train_loaders)][0])
            for padded_text, attention_masks, labels in train_loaders[b%len(train_loaders)][0]:
               if torch.cuda.is_available():
                  padded_text, attention_masks, labels = padded_text.cuda(), attention_masks.cuda(), labels.cuda()
            
               bias_optimizers[b].zero_grad()
               bias_output = bias_model(padded_text, attention_masks)[0]
                  
               iter += 1
               
               loss = CE_loss(F.softmax(bias_output, dim=1), labels)
               loss.backward()
               
               clip_grad_norm_(bias_model.parameters(), max_norm=1)
               bias_optimizers[b].step()
               bias_schedulers[b].step()
               total_loss += loss.item()
               
               if iter>args.pretrain_iters:
                  break
               
         print('*' * 89)
         word_asr, _ = evaluation(word_test_loader_poison_pretrain, bias_model)
         sent_asr, _ = evaluation(sent_test_loader_poison_pretrain, bias_model)
         syn_asr, _ = evaluation(syn_test_loader_poison_pretrain, bias_model)
         clean_acc, _ = evaluation(test_loader_clean, bias_model)
         if args.num_bias_experts>3 and args.style_poison_data_path!=None:
            style_asr, _ = evaluation(style_test_loader_poison_pretrain, bias_model)
            print(f'*** pretrain bias model {b} asrs ***', '(word_asr, sent_asr, syn_asr, style_asr, clean_acc)', word_asr, sent_asr, syn_asr, style_asr, clean_acc)
            write_results([f'*** pretrain bias model {b} asrs ***', '(word_asr, sent_asr, syn_asr, style_asr, clean_acc)', word_asr, sent_asr, syn_asr, style_asr, clean_acc])
         else:
            print(f'*** pretrain bias model {b} asrs ***', '(word_asr, sent_asr, syn_asr, clean_acc)', word_asr, sent_asr, syn_asr, clean_acc)
            write_results([f'*** pretrain bias model {b} asrs ***', '(word_asr, sent_asr, syn_asr, clean_acc)', word_asr, sent_asr, syn_asr, clean_acc])
         if args.save_pretrained!='':
            bias_model.module.save_pretrained(os.path.join(args.save_pretrained, f'{train_loaders[b%len(train_loaders)][1]}_bias_model_{num_hidden_layers[b]}_layers_{EPOCHS_PRETRAIN}_epoch'), from_pt=True) 
   except KeyboardInterrupt:
      print('-' * 89)
      print('Exiting from pretraining early')
      
   if args.num_bias_experts==3 and args.pretrain_gate:
      if args.gate_on_confidence:
         print('ERROR: pretraining gate on confidence not implemented')
         exit()
      else:
         try:
            print('start pretraining gate model')
            write_results(['start pretraining gate'])
            flag = 0
            total_loss = 0
            CE_loss = nn.CrossEntropyLoss()
            for epoch in range(EPOCHS_PRETRAIN):
               iter = 0
               gate_model.train()
               NUM_EX = len(gate_train_loader)
               for padded_text, attention_masks, labels in tqdm(gate_train_loader):
                  if torch.cuda.is_available():
                     padded_text, attention_masks, labels = padded_text.cuda(), attention_masks.cuda(), labels.cuda()
                  
                  gate_output = gate_model(padded_text, attention_masks)[0]
                  gate_output = F.softmax(gate_output, dim=1)
                  loss = CE_loss(gate_output, labels)
      
                  optimizer_gate.zero_grad()
                  loss.backward()
                  
                  clip_grad_norm_(gate_model.parameters(), max_norm=1)
                  optimizer_gate.step()
                  scheduler_gate.step()
                  
                  if iter>args.pretrain_iters:
                     break
                  
               print('*' * 89)
            gate_acc = evaluation(gate_test_loader, gate_model)
            print('gate_acc:', gate_acc)
            if args.save_pretrained!='':
               if args.with_clean_gate:
                  gate_model.module.save_pretrained(os.path.join(args.save_pretrained, f'with_clean_gate_model_{EPOCHS_PRETRAIN}_epoch'), from_pt=True) 
               else:
                  gate_model.module.save_pretrained(os.path.join(args.save_pretrained, f'no_clean_gate_model_{EPOCHS_PRETRAIN}_epoch'), from_pt=True) 
         except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from pretraining early')


def train():
   last_train_avg_loss = 1e10
   try:
      print('start training main model')
      write_results(['start training PoE'])
      flag = 0
      for epoch in range(EPOCHS):
         iter = 0
         model.train()
         total_loss = 0
         gs = torch.empty((100,args.batch_size), dtype=torch.float64)
         g_count = 0
         NUM_EX = len(train_loader_poison)
         times=[]
         for padded_text, attention_masks, labels in tqdm(train_loader_poison):
            tic = time.process_time()
            if args.train_iters!=-1 and iter>args.train_iters:
               break
         
            if torch.cuda.is_available():
               padded_text, attention_masks, labels = padded_text.cuda(), attention_masks.cuda(), labels.cuda()
            
            o = model(padded_text, attention_masks)
            output = o[0]
            last_hidden = o[1][len(o[1])-1]
            bias_outputs = torch.empty((args.num_bias_experts, padded_text.size()[0], num_classes), dtype=torch.float64)
            if torch.cuda.is_available():
               bias_outputs = bias_outputs.cuda()
            for i, bias_model in enumerate(bias_models):
               bias_outputs[i] = bias_model(padded_text, attention_masks)[0]
               
                        
            gate_output=[]
            if not args.no_gate:
               if args.gate_on_confidence:
                  bo = torch.flatten(bias_outputs.detach().permute(1,0,2), start_dim=1).float()
                  gate_output = gate_model(bo)
                  
                  _, idx = gate_output.topk(args.top_k_routing)
                  mask = torch.zeros_like(gate_output)
                  for i,j in enumerate(idx):
                     mask[i][j] = 1
                  gate_output *= mask
                  
                  gate_output = F.softmax(gate_output, dim=1)
               elif args.gate_on_hidden:
                  lh = last_hidden.detach().clone()
                  gate_output = gate_model(lh[:,0,:])
                  
                  _, idx = gate_output.topk(args.top_k_routing) 
                  mask = torch.zeros_like(gate_output)
                  for i,j in enumerate(idx):
                     mask[i][j] = 1
                  gate_output *= mask
                  
                  gate_output = F.softmax(gate_output, dim=1)
               else:
                  gate_output = gate_model(padded_text, attention_masks)[0]
                  
                  _, idx = gate_output.topk(args.top_k_routing)
                  mask = torch.zeros_like(gate_output)
                  for i,j in enumerate(idx):
                     mask[i][j] = 1
                  gate_output *= mask
                  
                  gate_output = F.softmax(gate_output, dim=1)
               
            iter += 1
            
            loss = 0
            if args.do_Rdrop:
               output_3 = model(padded_text, attention_masks)[0]

               loss_2 = kl_loss(output, output_3)
               if args.rdrop_mode_1:
                  loss_1 = 0.5 * (poe_loss(output, bias_outputs, gate_output, labels) + poe_loss(output_3, bias_outputs, gate_output, labels))
                  loss = loss_1 + rdrop_alpha * loss_2
               elif args.rdrop_mode_2:
                  loss = poe_loss(output, bias_outputs, gate_output, labels) + rdrop_alpha * loss_2
               else:
                  print('ERROR: please select r drop mode')
                  exit()
            else:
               loss = poe_loss(output, bias_outputs, gate_output, labels)
               
            optimizer.zero_grad()
            for optimizer_bias in bias_optimizers:
               optimizer_bias.zero_grad()
            if not args.no_gate:
               optimizer_gate.zero_grad()
            loss.backward()
            
            clip_grad_norm_(model.parameters(), max_norm=1)
            if not args.no_gate:
               clip_grad_norm_(gate_model.parameters(), max_norm=1)
            for bias_model in bias_models:
               clip_grad_norm_(bias_model.parameters(), max_norm=1)
               
            if args.freeze_main_iters!=-1 and (epoch!=0 or iter>args.freeze_main_iters):
               optimizer.step()
               scheduler.step()
            if args.freeze_bias_iters!=-1 and (epoch!=0 or iter>args.freeze_bias_iters):  
               for i, optimizer_bias in enumerate(bias_optimizers):
                  optimizer_bias.step()
                  bias_schedulers[i].step()
            if not args.no_gate:
               optimizer_gate.step()
               scheduler_gate.step()
            total_loss += loss.item()
            
            toc = time.process_time()
            times.append(toc-tic)
            if args.time and iter>=100:
               times = np.array(times)
               low, high = st.t.interval(alpha=0.95, df=len(times)-1, loc=np.mean(times), scale=st.sem(times))
               print('time', np.mean(times), np.mean(times)-low, (low, high))
               exit()
      
         print('*' * 89)
   
      print("word_poison_path_pretrain:", args.word_poison_path_pretrain)
      print("sent_poison_path_pretrain:", args.sent_poison_path_pretrain)
      if args.num_bias_experts>2:
         print("syn_poison_path_pretrain:", args.syn_poison_path_pretrain)
      if args.num_bias_experts>3 and args.style_poison_data_path!=None:
         print("style_poison_path_pretrain:", args.style_poison_path_pretrain)
      if args.do_Rdrop:
         if args.rdrop_mode_1:
            print('rdrop mode: 1')
         elif args.rdrop_mode_2:
            print('rdrop mode: 2')
         else:
            print('ERROR: rdrop mode not selected')
            exit()
      else:
         print('do_Rdrop: False')
      write_results(["**poison_path**", args.poison_data_path])
      write_results(["**word_poison_path_pretrain**", args.word_poison_path_pretrain])
      write_results(["**sent_poison_path_pretrain**", args.sent_poison_path_pretrain])
      if args.num_bias_experts>2:
         write_results(["**syn_poison_path_pretrain**", args.syn_poison_path_pretrain])
      if args.num_bias_experts>3 and args.style_poison_data_path!=None:
         write_results(["style_poison_path_pretrain:", args.style_poison_path_pretrain])
         
      write_results(['gate hidden layers', gate_hidden_layers])
      final_poison_success_rate_test, main_poison_logits = evaluation(test_loader_poison, model)
      final_clean_acc_test, main_clean_logits = evaluation(test_loader_clean, model)
      final_poison_success_rate_dev, _ = evaluation(dev_loader_poison, model)
      final_clean_acc_dev, _ = evaluation(dev_loader_clean, model)
      write_results(['*** final result ***', '(final_poison_success_rate_dev, final_clean_acc_dev, final_poison_success_rate_test, final_clean_acc_test)', final_poison_success_rate_dev, final_clean_acc_dev, final_poison_success_rate_test, final_clean_acc_test])
      print('*** final result ***', '(final_poison_success_rate_dev, final_clean_acc_dev, final_poison_success_rate_test, final_clean_acc_test)', final_poison_success_rate_dev, final_clean_acc_dev, final_poison_success_rate_test, final_clean_acc_test)
      if final_poison_success_rate_dev in dev_asrs:
         dev_asrs[final_poison_success_rate_dev].append((poe_alpha, num_hidden_layers, gate_hidden_layers, tune_iter))
      else:
         dev_asrs[final_poison_success_rate_dev] = [(poe_alpha, num_hidden_layers, gate_hidden_layers, tune_iter)]
      if final_poison_success_rate_test in test_asrs:
         test_asrs[final_poison_success_rate_test].append((poe_alpha, num_hidden_layers, gate_hidden_layers, tune_iter))
      else:
         test_asrs[final_poison_success_rate_test] = [(poe_alpha, num_hidden_layers, gate_hidden_layers, tune_iter)]
      
      asrs.append(final_poison_success_rate_test)
      accs.append(final_clean_acc_test)

               
      word_asr, main_word_logits = evaluation(word_test_loader_poison, model)
      sent_asr, main_sent_logits = evaluation(sent_test_loader_poison, model)
      syn_asr, main_syn_logits = evaluation(syn_test_loader_poison, model)
      if args.num_bias_experts>3 and args.style_poison_data_path!=None:
         style_asr, main_style_logits = evaluation(style_test_loader_poison, model)
         write_results([f'*** final main model asrs ***', '(word_asr, sent_asr, syn_asr, style_asr)', word_asr, sent_asr, syn_asr, style_asr])
         print(f'*** final main model asrs ***', '(word_asr, sent_asr, syn_asr, style_asr)', word_asr, sent_asr, syn_asr, style_asr)
      else:
         write_results([f'*** final main model asrs ***', '(word_asr, sent_asr, syn_asr)', word_asr, sent_asr, syn_asr])
         print(f'*** final main model asrs ***', '(word_asr, sent_asr, syn_asr)', word_asr, sent_asr, syn_asr)
      
      bias_word_logits=[]
      bias_sent_logits=[]
      bias_syn_logits=[]
      bias_style_logits=[]
      bias_clean_logits=[]
      for i, bias_model in enumerate(bias_models):
         word_asr, bias_word_logits_t = evaluation(word_test_loader_poison, bias_model)
         sent_asr,  bias_sent_logits_t = evaluation(sent_test_loader_poison, bias_model)
         syn_asr, bias_syn_logits_t = evaluation(syn_test_loader_poison, bias_model)
         clean_acc, bias_clean_logits_t = evaluation(test_loader_clean, bias_model)
         
         bias_word_logits.append(bias_word_logits_t)
         bias_sent_logits.append(bias_sent_logits_t)
         bias_syn_logits.append(bias_syn_logits_t)
         bias_clean_logits.append(bias_clean_logits_t)
         
         if args.num_bias_experts>3 and args.style_poison_data_path!=None:
            style_asr, bias_style_logits_t = evaluation(style_test_loader_poison, bias_model)
            bias_style_logits.append(bias_style_logits_t)
         
         if args.num_bias_experts>3 and args.style_poison_data_path!=None:
            write_results([f'*** final bias model {i} asrs ***', word_asr, sent_asr, syn_asr, style_asr, clean_acc])
            print(f'*** final bias model {i} asrs ***', word_asr, sent_asr, syn_asr, style_asr, clean_acc)
         else:
            write_results([f'*** final bias model {i} asrs ***', word_asr, sent_asr, syn_asr, clean_acc])
            print(f'*** final bias model {i} asrs ***', word_asr, sent_asr, syn_asr, clean_acc)
         
      word_asr, bias_word_logit = evaluation_whole_bias(word_test_loader_poison, bias_models)
      sent_asr,  bias_sent_logit = evaluation_whole_bias(sent_test_loader_poison, bias_models)
      syn_asr, bias_syn_logit = evaluation_whole_bias(syn_test_loader_poison, bias_models)
      clean_acc, bias_clean_logit = evaluation_whole_bias(test_loader_clean, bias_models)
      if args.num_bias_experts>3 and args.style_poison_data_path!=None:
         style_asr, bias_style_logit = evaluation_whole_bias(style_test_loader_poison, bias_models)
         write_results([f'*** final bias models ensembled asrs ***', word_asr, sent_asr, syn_asr, style_asr, clean_acc])
         print(f'*** final bias models ensembled asrs ***', word_asr, sent_asr, syn_asr, style_asr, clean_acc)
      else:
         write_results([f'*** final bias models ensembled asrs ***', word_asr, sent_asr, syn_asr, clean_acc])
         print(f'*** final bias models ensembled asrs ***', word_asr, sent_asr, syn_asr, clean_acc)
      
      
      
      clean_small_acc, _ = evaluation(clean_small_loader, model)
      print('*** clean small (pseudo dev) accuracy ***', clean_small_acc)
      write_results(['*** clean small (pseudo dev) accuracy ***', clean_small_acc])
      print('*** final result ***', final_poison_success_rate_dev, final_clean_acc_dev, final_poison_success_rate_test, final_clean_acc_test)
      
      if args.num_bias_experts==3 and not args.gate_on_hidden and not args.gate_on_confidence and args.gate_data_path!='':
         gate_acc, _ = evaluation(gate_test_loader, gate_model)
         print('gate_acc:', gate_acc)
      
      try:
         total_correct_asr, total_selected_asr, total_correct_acc, total_selected_acc = evaluation_pseudo(train_loader_poison)
         loader_len = len(train_loader_poison)*args.batch_size
         write_results(['*** detected poison ***', '(detected, loader_len, ratio)', total_selected_asr, loader_len, total_selected_asr/loader_len])
         print('detected poison:', '(detected, loader_len, ratio)', total_selected_asr, loader_len, total_selected_asr/loader_len)
         if (clean_small_acc>0.8 and args.data=='offenseval') or (clean_small_acc>0.9 and 'trec' in args.data) or (clean_small_acc>0.85 and 'sst' in args.data):
            detect[total_selected_asr/loader_len] = (poe_alpha, num_hidden_layers, gate_hidden_layers, rdrop_alpha, tune_iter)
         if total_selected_asr>int(0.03*len(train_loader_poison)):
            pseudo_dev_asr = total_correct_asr / total_selected_asr
         else:
            pseudo_dev_asr = -1
         if total_selected_acc>int(0.1*len(train_loader_poison)):
            pseudo_dev_acc = total_correct_acc / total_selected_acc
         else:
            pseudo_dev_acc = -1
         write_results(['*** pseudo ***', ' pseudo_dev_asr', pseudo_dev_asr, ' pseudo_dev_acc', pseudo_dev_acc])
         print('*** pseudo ***', ' pseudo_dev_asr', pseudo_dev_asr, ' pseudo_dev_acc', pseudo_dev_acc)
         if pseudo_dev_asr in pseudo_dev_asrs:
            pseudo_dev_asrs[pseudo_dev_asr].append((poe_alpha, num_hidden_layers, gate_hidden_layers, rdrop_alpha, tune_iter)) 
         else:
            pseudo_dev_asrs[pseudo_dev_asr] = [(poe_alpha, num_hidden_layers, gate_hidden_layers, rdrop_alpha, tune_iter)]
         
         # for convenience when viewing results
         CLEAN_SMALL_ACC_CUTOFF = 0.8    # set as desired
         if total_selected_asr/loader_len>0.3 and total_selected_asr/loader_len<0.5 and clean_small_acc>CLEAN_SMALL_ACC_CUTOFF:
            if pseudo_dev_asr in pseudo_dev_good_asrs:
               pseudo_dev_good_asrs[pseudo_dev_asr].append((poe_alpha, num_hidden_layers, gate_hidden_layers, rdrop_alpha, tune_iter))
            else: 
               pseudo_dev_good_asrs[pseudo_dev_asr] = [(poe_alpha, num_hidden_layers, gate_hidden_layers, rdrop_alpha, tune_iter)]
            
      except:
         write_results(['*** pseudo ***', ' ERROR ENCOUNTERED WHILE PROCESSING PSEUDO DEV RESULTS'])
         print('*** pseudo ***', ' ERROR ENCOUNTERED WHILE PROCESSING PSEUDO DEV RESULTS')
          
                  
      df_word = pd.DataFrame({'main_word_logits': main_word_logits})
      df_sent = pd.DataFrame({'main_sent_logits': main_sent_logits})
      df_syn = pd.DataFrame({'main_syn_logits': main_syn_logits})
      if args.num_bias_experts>3 and args.style_poison_data_path!=None:
         df_style = pd.DataFrame({'main_style_logits': main_style_logits})
      df_clean = pd.DataFrame({'main_clean_logits': main_clean_logits})
      df_poison = pd.DataFrame({'main_poison_logits': main_poison_logits})
      
      df_word['bias_word_logits'] = bias_word_logit
      df_sent['bias_sent_logits'] = bias_sent_logit
      df_syn['bias_syn_logits'] = bias_syn_logit
      if args.num_bias_experts>3 and args.style_poison_data_path!=None:
         df_style['bias_style_logits'] = bias_style_logit
      df_clean['bias_clean_logits'] = bias_clean_logit
         
      for i in range(len(bias_sent_logits)):
         df_word[f'bias_word_logits_{i}'] = bias_word_logits[i]
         df_sent[f'bias_sent_logits_{i}'] = bias_sent_logits[i]
         df_syn[f'bias_syn_logits_{i}'] = bias_syn_logits[i]
         if args.num_bias_experts>3 and args.style_poison_data_path!=None:
            df_style[f'bias_style_logits_{i}'] = bias_style_logits[i]
         df_clean[f'bias_clean_logits_{i}'] = bias_clean_logits[i]
      
      
      os.makedirs(os.path.join('imgs', img_dir), exist_ok=True)
      sns.histplot(data=df_word, x=f'bias_word_logits', binwidth=0.05, binrange=(0,1), label=f'bias models ensembled')
      p = sns.histplot(data=df_word, x='main_word_logits', binwidth=0.05, binrange=(0,1), label='main model')
      plt.legend()
      plt.xlabel('confidence')
      fig = p.get_figure()
      fig.savefig(os.path.join('imgs', img_dir, f'WordLogits_PoE{poe_alpha}_Rdrop{rdrop_alpha}_HiddenLayers{args.num_hidden_layers}_GateLayers{gate_hidden_layers}_{args.result_file}_{args.num_bias_experts}BiasExperts.png'))
      plt.clf()
      
      
      sns.histplot(data=df_sent, x=f'bias_sent_logits', binwidth=0.05, binrange=(0,1), label=f'bias models ensembled')
      p = sns.histplot(data=df_sent, x='main_sent_logits', binwidth=0.05, binrange=(0,1), label='main model')
      plt.legend()
      plt.xlabel('confidence')
      fig = p.get_figure()
      fig.savefig(os.path.join('imgs', img_dir, f'SentLogits_PoE{poe_alpha}_Rdrop{rdrop_alpha}_HiddenLayers{args.num_hidden_layers}_GateLayers{gate_hidden_layers}_{args.result_file}_{args.num_bias_experts}BiasExperts.png'))
      plt.clf()
      
      
      sns.histplot(data=df_syn, x=f'bias_syn_logits', binwidth=0.05, binrange=(0,1), label=f'bias models ensembled')
      p = sns.histplot(data=df_syn, x='main_syn_logits', binwidth=0.05, binrange=(0,1), label='main model')
      plt.legend()
      plt.xlabel('confidence')
      fig = p.get_figure()
      fig.savefig(os.path.join('imgs', img_dir, f'SynLogits_PoE{poe_alpha}_Rdrop{rdrop_alpha}_HiddenLayers{args.num_hidden_layers}_GateLayers{gate_hidden_layers}_{args.result_file}_{args.num_bias_experts}BiasExperts.png'))
      plt.clf()
      
      if args.num_bias_experts>3 and args.style_poison_data_path!=None:
         sns.histplot(data=df_style, x=f'bias_style_logits', binwidth=0.05, binrange=(0,1), label=f'bias models ensembled')
         p = sns.histplot(data=df_style, x='main_style_logits', binwidth=0.05, binrange=(0,1), label='main model')
         plt.legend()
         plt.xlabel('confidence')
         fig = p.get_figure()
         fig.savefig(os.path.join('imgs', img_dir, f'StyleLogits_PoE{poe_alpha}_Rdrop{rdrop_alpha}_HiddenLayers{args.num_hidden_layers}_GateLayers{gate_hidden_layers}_{args.result_file}_{args.num_bias_experts}BiasExperts.png'))
         plt.clf()
      
      
      sns.histplot(data=df_clean, x=f'bias_clean_logits', binwidth=0.05, binrange=(0,1), label=f'bias models ensembled')
      p = sns.histplot(data=df_clean, x='main_clean_logits', binwidth=0.05, binrange=(0,1), label='main model')
      plt.legend()
      plt.xlabel('confidence')
      fig = p.get_figure()
      fig.savefig(os.path.join('imgs', img_dir, f'CleanLogits_PoE{poe_alpha}_Rdrop{rdrop_alpha}_HiddenLayers{args.num_hidden_layers}_GateLayers{gate_hidden_layers}_{args.result_file}_{args.num_bias_experts}BiasExperts.png'))
      plt.clf()
      
      
      
   except KeyboardInterrupt:
      print('-' * 89)
      print('Exiting from training early')
      
    
    
def plain_train():
   last_train_avg_loss = 1e10
   try:
      print('start training')
      write_results(['start training'])
      flag = 0
      for epoch in range(EPOCHS):
         iter = 0
         model.train()
         total_loss = 0
         g_count = 0
         NUM_EX = len(train_loader_poison)
         times=[]
         for padded_text, attention_masks, labels in tqdm(train_loader_poison):
            tic = time.process_time()
            if args.train_iters!=-1 and iter>args.train_iters:
               break
         
            if torch.cuda.is_available():
               padded_text, attention_masks, labels = padded_text.cuda(), attention_masks.cuda(), labels.cuda()
         
            
            output = model(padded_text, attention_masks)[0]            
               
            iter += 1
            
            loss = 0
            if args.do_Rdrop:
               output_3 = model(padded_text, attention_masks)[0]

               loss_2 = kl_loss(output, output_3)
               if args.rdrop_mode_2:
                  loss = plain_loss(output, labels) + args.rdrop_alpha * loss_2
               else:
                  print('ERROR: please select r drop mode 2')
                  exit()
            else:
               loss = plain_loss(output, labels)
               
            optimizer.zero_grad()
            loss.backward()
            
            clip_grad_norm_(model.parameters(), max_norm=1)
               
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()    
            
            toc = time.process_time()
            times.append(toc-tic)
            if args.time and iter>=100:
               times = np.array(times)
               low, high = st.t.interval(alpha=0.95, df=len(times)-1, loc=np.mean(times), scale=st.sem(times))
               print('times', np.mean(times), np.mean(times)-low, (low, high))
               exit()  
      
         print('*' * 89)
   
      print("poison_data_path:", args.poison_data_path)
      print("word_poison_data_path:", args.word_poison_data_path)
      print("sent_poison_data_path:", args.sent_poison_data_path)
      print("syn_poison_data_path:", args.syn_poison_data_path)
      if args.style_poison_data_path!=None:
         print("style_poison_data_path:", args.style_poison_data_path)
      if args.do_Rdrop:
         if args.rdrop_mode_1:
            print('rdrop mode: 1')
         elif args.rdrop_mode_2:
            print('rdrop mode: 2')
         else:
            print('ERROR: rdrop mode not selected')
            exit()
      else:
         print('do_Rdrop: False')
         
      final_poison_success_rate_test, main_poison_logits = evaluation(test_loader_poison, model)
      final_clean_acc_test, main_clean_logits = evaluation(test_loader_clean, model)
      final_poison_success_rate_dev, _ = evaluation(dev_loader_poison, model)
      final_clean_acc_dev, _ = evaluation(dev_loader_clean, model)
      write_results(['*** final result ***', '(final_poison_success_rate_dev, final_clean_acc_dev, final_poison_success_rate_test, final_clean_acc_test)', final_poison_success_rate_dev, final_clean_acc_dev, final_poison_success_rate_test, final_clean_acc_test])
      print('*** final result ***', '(final_poison_success_rate_dev, final_clean_acc_dev, final_poison_success_rate_test, final_clean_acc_test)', final_poison_success_rate_dev, final_clean_acc_dev, final_poison_success_rate_test, final_clean_acc_test)
         
      word_asr, main_word_logits = evaluation(word_test_loader_poison, model)
      sent_asr, main_sent_logits = evaluation(sent_test_loader_poison, model)
      syn_asr, main_syn_logits = evaluation(syn_test_loader_poison, model)
      if args.style_poison_data_path!=None:
         style_asr, main_style_logits = evaluation(style_test_loader_poison, model)
         write_results([f'*** final main model asrs ***', '(word_asr, sent_asr, syn_asr, style_asr)', word_asr, sent_asr, syn_asr, style_asr])
         print(f'*** final main model asrs ***', '(word_asr, sent_asr, syn_asr, style_asr)', word_asr, sent_asr, syn_asr, style_asr)
      else:
         write_results([f'*** final main model asrs ***', '(word_asr, sent_asr, syn_asr)', word_asr, sent_asr, syn_asr])
         print(f'*** final main model asrs ***', '(word_asr, sent_asr, syn_asr)', word_asr, sent_asr, syn_asr)
         
      df_word = pd.DataFrame({'main_word_logits': main_word_logits})
      df_sent = pd.DataFrame({'main_sent_logits': main_sent_logits})
      df_syn = pd.DataFrame({'main_syn_logits': main_syn_logits})
      if args.style_poison_data_path!=None:
         df_style = pd.DataFrame({'main_style_logits': main_style_logits})
      df_clean = pd.DataFrame({'main_clean_logits': main_clean_logits})
      df_poison = pd.DataFrame({'main_poison_logits': main_poison_logits})
      
               
      
      p = sns.histplot(data=df_word, x='main_word_logits', binwidth=0.05, binrange=(0,1), label='main model')
      plt.legend()
      plt.xlabel('confidence')
      fig = p.get_figure()
      fig.savefig(os.path.join('imgs', f'WordLogits_{args.result_file}_{args.num_bias_experts}BiasExperts.png'))
      plt.clf()
      
      p = sns.histplot(data=df_sent, x='main_sent_logits', binwidth=0.05, binrange=(0,1), label='main model')
      plt.legend()
      plt.xlabel('confidence')
      fig = p.get_figure()
      fig.savefig(os.path.join('imgs', f'SentLogits_{args.result_file}_{args.num_bias_experts}BiasExperts.png'))
      plt.clf()
      
      p = sns.histplot(data=df_syn, x='main_syn_logits', binwidth=0.05, binrange=(0,1), label='main model')
      plt.legend()
      plt.xlabel('confidence')
      fig = p.get_figure()
      fig.savefig(os.path.join('imgs', f'SynLogits_{args.result_file}_{args.num_bias_experts}BiasExperts.png'))
      plt.clf()
      
      if args.style_poison_data_path!=None:
         p = sns.histplot(data=df_style, x='main_style_logits', binwidth=0.05, binrange=(0,1), label='main model')
         plt.legend()
         plt.xlabel('confidence')
         fig = p.get_figure()
         fig.savefig(os.path.join('imgs', f'StyleLogits_{args.result_file}_{args.num_bias_experts}BiasExperts.png'))
         plt.clf()
      
      p = sns.histplot(data=df_clean, x='main_clean_logits', binwidth=0.05, binrange=(0,1), label='main model')
      plt.legend()
      plt.xlabel('confidence')
      fig = p.get_figure()
      fig.savefig(os.path.join('imgs', f'CleanLogits_{args.result_file}_{args.num_bias_experts}BiasExperts.png'))
      plt.clf()
      
      
   except KeyboardInterrupt:
      print('-' * 89)
      print('Exiting from training early')



def write_results(result):
   with open(result_file, 'a') as file:
      csv_writer = csv.writer(file)
      csv_writer.writerow(result)

def write_results_small(result):
   with open(result_file_small, 'a') as file:
      csv_writer = csv.writer(file)
      csv_writer.writerow(result)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
   """ Create a schedule with a learning rate that decreases linearly after
   linearly increasing during a warmup period.

   From:
       https://github.com/uds-lsv/bert-stable-fine-tuning/blob/master/src/transformers/optimization.py
   """

   def lr_lambda(current_step):
      if current_step < num_warmup_steps:
         return float(current_step) / float(max(1, num_warmup_steps))
      return max(
         0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
         )

   return LambdaLR(optimizer, lr_lambda, last_epoch)


def initialize_bert_model(model):
   for module in model.modules():
      if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
         module.weight.data.normal_(mean=0.0, std=0.02)
      elif isinstance(module, torch.nn.LayerNorm):
         module.bias.data.zero_()
         module.weight.data.fill_(1.0)
   return model


def set_seed(seed: int):
   """Sets the relevant random seeds."""
   random.seed(seed)
   np.random.seed(seed)
   torch.random.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--data', type=str, default='sst-2')
   parser.add_argument('--batch_size', type=int, default=32)
   # args for optimizer
   parser.add_argument('--lr', type=float, default=2e-5)
   parser.add_argument('--small_lr', type=float, default=5e-4)
   parser.add_argument('--weight_decay', default=1e-2, type=float)  # BERT default
   parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")  # BERT default
   parser.add_argument("--warmup_ratio", default=0.1, type=float, help="Linear warmup over warmup_steps.")  # BERT default
   parser.add_argument('--bias_correction', default=True)
   # args for training
   parser.add_argument('--model_name', type=str, default='bert-base-uncased')
   parser.add_argument('--epoch', type=int, default=10)
   parser.add_argument('--seed', type=int, default=42)
   parser.add_argument('--small_model_path', default="")
   parser.add_argument("--small_model_name", default="")
   parser.add_argument("--do_reinit", type=bool, default=False)
   # args for poison
   parser.add_argument('--poison_rate', type=int, default=20)
   parser.add_argument('--clean_data_path', )
   parser.add_argument('--poison_data_path',)
   parser.add_argument('--save_path', default='')
   parser.add_argument('--gpu', type=int, default=2)
   parser.add_argument('--num_bias_layers', type=int, default=3)
   parser.add_argument('--do_PoE', type=bool, default=True, help="If selected, train model with PoE")
   parser.add_argument('--poe_alpha', type=float, default=1.0)
   parser.add_argument('--do_Rdrop', type=bool, default=False)
   parser.add_argument('--dropout_prob', type=float, default=0.1)
   parser.add_argument('--rdrop_alpha', type=float, default=1.0)  #1-30 increments of 5 - usually 15
   parser.add_argument('--rdrop_mode_1', type=bool, default=False)
   parser.add_argument('--rdrop_mode_2', type=bool, default=False)
   parser.add_argument('--temperature', type=float, default=1.0)
   parser.add_argument('--result_path', default='results')
   parser.add_argument('--ensemble_layer_num', type=int, default=0)
   
   # other args
   parser.add_argument('--num_bias_experts', type=int, default=3)
   parser.add_argument('--reg_weight', type=float, default=0)
   parser.add_argument("--num_hidden_layers", type=str, default='3')
   parser.add_argument("--gate_hidden_layers", type=int, default=2)
   parser.add_argument('--word_poison_data_path',)
   parser.add_argument('--sent_poison_data_path',)
   parser.add_argument('--syn_poison_data_path',)
   parser.add_argument('--style_poison_data_path',)
   parser.add_argument('--kl_div', type=bool, default=False)
   parser.add_argument('--no_gate', type=bool, default=False)
   parser.add_argument('--save_pretrained', default='')
   parser.add_argument('--epoch_pretrain', type=int, default=1)
   parser.add_argument('--freeze_bias_iters', type=int, default=-1)
   parser.add_argument('--pretrain_iters', type=int, default=800)
   parser.add_argument('--with_clean_gate', type=bool, default=False)
   parser.add_argument('--gate_data_path', default='')
   parser.add_argument('--word_poison_path_pretrain',)
   parser.add_argument('--sent_poison_path_pretrain',)
   parser.add_argument('--syn_poison_path_pretrain',)
   parser.add_argument('--style_poison_path_pretrain',)
   parser.add_argument('--pretrain_gate', type=bool, default=False)
   parser.add_argument('--top_k_routing', type=int, default=-1)
   
   # high -> ASR, low -> Acc
   parser.add_argument("--pseudo_dev_thre_high", type=float, default=0.8) # or higher such as 1 or 0.98
   parser.add_argument("--pseudo_dev_thre_low", type=float, default=0.7) # since clean data can be easy
   parser.add_argument("--thre_main_high", type=float, default=0.6) # 0.6-0.7, can even set to 0.8 since easy data should have very high confidence 
   parser.add_argument("--thre_main_low", type=float, default=0.8) # or lower such as 0.4-0.6 (main usually has very low confidence on poisoned data)
   parser.add_argument('--result_file', default='')
   parser.add_argument('--gate_on_confidence', type=bool, default=False)
   parser.add_argument('--gate_on_hidden', type=bool, default=False)
   parser.add_argument("--type", type=str, default='test', help="test, conf, tune1-4, or ana1-3")
   parser.add_argument('--train_iters', type=int, default=-1)
   parser.add_argument("--img_dir", type=str, default='test')
   parser.add_argument('--tune_iters', type=int, default=20)   # number of settings to test for randomized hyperparameter search (tune3)
   parser.add_argument('--freeze_main_iters', type=int, default=0)
   parser.add_argument('--ablate_trigger', default='')
   parser.add_argument('--time', type=bool, default=False)  # get time estimates
   
   
   args = parser.parse_args()
   data_selected = args.data
   BATCH_SIZE = args.batch_size
   weight_decay = args.weight_decay
   lr = args.lr
   EPOCHS = args.epoch
   EPOCHS_PRETRAIN = args.epoch_pretrain
   if args.gate_on_hidden and args.gate_on_confidence:
      print('ERROR: select at most 1 gate type')
      exit()
      
   img_dir = args.img_dir
   if args.img_dir!='test' and args.img_dir!='other':
      img_dir = os.path.join(args.img_dir, args.data, str(args.num_bias_experts)+'poison')
   
   os.makedirs(os.path.join(args.result_path, args.small_model_name), exist_ok=True)
   if args.result_file!='':
      result_file = os.path.join(args.result_path, args.small_model_name, args.result_file+f'_{args.num_bias_experts}BiasExperts.csv')
   
   pseudo_dev_asrs = {}
   dev_asrs = {}
   test_asrs = {}
   pseudo_dev_good_asrs = {}
   detect = {}
   
   num_classes=4 if 'ag' in data_selected else 2
   if 'trec' in data_selected:
      num_classes=6
   
   if args.num_bias_experts==0:
      seeds = [args.seed]
      asrs = []
      accs = []
      if args.type=='conf':
         random.seed(args.seed)
         seeds=[]
         for i in range(5):
            seeds.append(random.randint(0, 2**32 - 1))
      
      print("num_epoch:", args.epoch)
      print("num_bias_experts:", args.num_bias_experts)
      if args.do_Rdrop:
         if args.rdrop_mode_1:
            print('rdrop mode: 1')
         elif args.rdrop_mode_2:
            print('rdrop mode: 2')
         else:
            print('ERROR: rdrop mode not selected')
            exit()
      else:
         print('do_Rdrop: False')
      
      print("poison_data_path:", args.poison_data_path)
      print("word_poison_data_path:", args.word_poison_data_path)
      print("sent_poison_data_path:", args.sent_poison_data_path)
      print("syn_poison_data_path:", args.syn_poison_data_path)
      if args.style_poison_data_path!=None:
         print("style_poison_data_path:", args.style_poison_data_path)
      
      write_results(["num_epoch:", args.epoch])
      write_results(["num_bias_experts:", args.num_bias_experts])
      if args.do_Rdrop:
         if args.rdrop_mode_1:
            write_results(['rdrop mode: 1'])
         elif args.rdrop_mode_2:
            write_results(['rdrop mode: 2'])
      else:
         write_results(['do_Rdrop: False'])
      
      write_results(["word_poison_data_path:", args.word_poison_data_path])
      write_results(["sent_poison_data_path:", args.sent_poison_data_path])
      write_results(["syn_poison_data_path:", args.syn_poison_data_path])
      if args.style_poison_data_path!=None:
         write_results(["style_poison_data_path:", args.style_poison_data_path])
      
      
      os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3' 
      torch.cuda.set_device(args.gpu)
      print('GPU:', torch.cuda.current_device())
      
      # load data
      clean_train_data, clean_dev_data, clean_test_data = get_all_data(args.clean_data_path)
      poison_train_data, poison_dev_data, poison_test_data = get_all_data(args.poison_data_path)
      packDataset_util = packDataset_util_bert()
      train_loader_poison = packDataset_util.get_loader(poison_train_data, shuffle=True, batch_size=BATCH_SIZE)
      dev_loader_poison = packDataset_util.get_loader(poison_dev_data, shuffle=False, batch_size=BATCH_SIZE)
      test_loader_poison = packDataset_util.get_loader(poison_test_data, shuffle=False, batch_size=BATCH_SIZE)
      train_loader_clean = packDataset_util.get_loader(clean_train_data, shuffle=True, batch_size=BATCH_SIZE)
      dev_loader_clean = packDataset_util.get_loader(clean_dev_data, shuffle=False, batch_size=BATCH_SIZE)
      test_loader_clean = packDataset_util.get_loader(clean_test_data, shuffle=False, batch_size=BATCH_SIZE)
   
      print('main sets loaded')
   
      ### don't use dev sets ###
      _, _, word_poison_data_test = get_all_data(args.word_poison_data_path)
      _, _, sent_poison_data_test = get_all_data(args.sent_poison_data_path)
      _, _, syn_poison_data_test = get_all_data(args.syn_poison_data_path)
      word_test_loader_poison = packDataset_util.get_loader(word_poison_data_test, shuffle=True, batch_size=BATCH_SIZE)
      sent_test_loader_poison = packDataset_util.get_loader(sent_poison_data_test, shuffle=True, batch_size=BATCH_SIZE)
      syn_test_loader_poison = packDataset_util.get_loader(syn_poison_data_test, shuffle=True, batch_size=BATCH_SIZE)
      
      if args.style_poison_data_path!=None:
         _, _, style_poison_data_test = get_all_data(args.style_poison_data_path)
         style_test_loader_poison = packDataset_util.get_loader(style_poison_data_test, shuffle=True, batch_size=BATCH_SIZE)
         
      clean_small_path = args.clean_data_path.replace('clean_data','clean_small')
      clean_small_train, _, _ = get_all_data(clean_small_path)
      # use clean subset of train for pseudo dev accuracy
      clean_small_loader = packDataset_util.get_loader(clean_small_train, shuffle=True, batch_size=BATCH_SIZE)
               
      print('special sets loaded')
      
      for seed in seeds:
         set_seed(seed)
                  
         # load model
         config = AutoConfig.from_pretrained(args.model_name, num_labels=num_classes, output_hidden_states=True)
         config.ensemble_layer_num = args.ensemble_layer_num
         config.output_hidden_states = True
         model = BertForSequenceClassification.from_pretrained(args.model_name, output_hidden_states=True, return_dict=True, num_labels=num_classes)
         if torch.cuda.is_available():
            print('cuda is available')
            model = nn.DataParallel(model.cuda(), device_ids=[args.gpu])
         else:
            print('no cuda')
      
         criterion = nn.CrossEntropyLoss()
         
      
         # Prepare optimizer and schedule (linear warmup and decay)
         no_decay = ["bias", "LayerNorm.weight"]
      
         optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            ]
      
         optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.lr,
            eps=args.adam_epsilon,
            correct_bias=args.bias_correction
            )
      
      
         # Use suggested learning rate scheduler
         num_training_steps = len(poison_train_data) * args.epoch // args.batch_size
         warmup_steps = num_training_steps * args.warmup_ratio
         scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)
         plain_train()
      
      if args.type=='conf':
         avg_asr = statistics.mean(asrs)
         avg_acc = statistics.mean(accs)
         sd_asr = statistics.pstdev(asrs)
         sd_acc = statistics.pstdev(accs)
         
         print("Average ASR:", avg_asr, "Std Dev:", sd_asr)   
         print("Average Acc:", avg_acc, "Std Dev:", sd_acc)   
         write_results(["Average ASR", avg_asr, " Std Dev", sd_asr])   
         write_results(["Average Acc", avg_acc, " Std Dev", sd_acc])        
      exit()
   
   settings=[]
   seeds = [args.seed]
   asrs = []
   accs = []
   if args.type=='test':
      settings.append((args.poe_alpha, args.num_hidden_layers, args.gate_hidden_layers, args.rdrop_alpha))
   elif args.type=='conf':
      settings.append((args.poe_alpha, args.num_hidden_layers, args.gate_hidden_layers, args.rdrop_alpha))
      random.seed(args.seed)
      seeds=[]
      for i in range(5):
         seeds.append(random.randint(0, 2**32 - 1))
   elif args.type=='tune':   # more complete hyperparameter search
      if args.do_Rdrop:
         rdrop = [1, 5, 10, 15, 20, 25, 30]
      else:
         rdrop = [1]
      alphas = [1.0, 1.5, 2.0, 0.5, 2.5, 3.0, 3.5, 4.0, 6.0, 8.0] 
      layers = ['2', '3', '4', '5', '6', '1', '7', '8', '9', '10', '11', '12']
      gate_layers = [2, 3, 4, 5, 6, 1, 7, 8, 9, 10, 11, 12]
      if args.num_bias_experts==1:
         gate_layers=[1] 
      for l in layers:
         for a in alphas: 
            for g in gate_layers:
               for r in rdrop:
                  settings.append((a, l, g, r))
   elif args.type=='tune2':   # partial hyperparameter search
      if args.do_Rdrop:
         rdrop = [1, 5, 15, 30]
      else:
         rdrop = [1]
      alphas = [1.0, 2.0, 3.0]  
      layers = ['2', '3', '4', '6', '5', '1'] 
      gate_layers = [3, 4]
      if args.num_bias_experts==1:
         gate_layers=[1] 
      for l in layers:
         for a in alphas: 
            for g in gate_layers:
               for r in rdrop:
                  settings.append((a, l, g, r))
   elif args.type=='tune3':   # randomized hyperparamter search
      settings=set()
      if args.do_Rdrop:
         rdrop = [1, 5, 10, 15, 20, 25, 30]
      else:
         rdrop = [1]
      alphas = [1.0, 1.5, 2.0, 0.5, 0.1, 2.5, 3.0, 3.5, 4.0, 6.0, 8.0, 10.0, 15.0]
      layers = ['2', '3', '4', '5', '6', '1', '7', '8', '9', '10', '11', '12']
      gate_layers = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1]
      if args.num_bias_experts==1:
         gate_layers=[1] 
      while len(settings)<args.tune_iters: 
         set = (random.sample(alphas, 1)[0], random.sample(layers, 1)[0], random.sample(gate_layers, 1)[0], random.sample(rdrop, 1)[0])
         if set not in settings:
            settings.add(set)
   elif args.type=='tune4':   # limited rdrop hyperparamter search
      if args.do_Rdrop:
         rdrop = [1, 5]
      else:
         rdrop = [1]
      alphas = [1.0, 1.5, 2.0, 0.5, 0.1, 2.5, 3.0, 3.5]
      layers = ['2', '3', '4', '5', '6', '1', '7'] 
      gate_layers = [2, 3, 4, 5, 6] 
      if args.num_bias_experts==1:
         gate_layers=[1] 
      for l in layers:
         for a in alphas: 
            for g in gate_layers:
               for r in rdrop:
                  settings.append((a, l, g, r))
   elif args.type=='ana1':      # analysis of alpha
      alphas = [1.0, 1.5, 2.0, 0.5, 2.5, 3.0, 3.5]
      if args.num_bias_experts==1:
        gate_layers=1
      else:
        gate_layers = args.gate_hidden_layers
      for a in alphas:
         settings.append((a, args.num_hidden_layers, gate_layers, args.rdrop_alpha))
   elif args.type=='ana2':      # analysis of layer count
      layers = ['2', '3', '4', '5', '6', '1', '7'] 
      if args.num_bias_experts==1:
        gate_layers=1
      else:
        gate_layers = args.gate_hidden_layers
      for l in layers:
         settings.append((args.poe_alpha, l, gate_layers, args.rdrop_alpha))
   elif args.type=='ana3':       # analysis of gate layer count
      gate_layers = [2, 3, 4, 5, 6] 
      if args.num_bias_experts==1:
        print('ERROR - num_bias_experts must be >1 for gate experiments')
        exit()
      for g in gate_layers:
         settings.append((args.poe_alpha, args.num_hidden_layers, g, args.rdrop_alpha))
   
   # load data
   clean_train_data, clean_dev_data, clean_test_data = get_all_data(args.clean_data_path)
   poison_train_data, poison_dev_data, poison_test_data = get_all_data(args.poison_data_path)
   packDataset_util = packDataset_util_bert()
   train_loader_poison = packDataset_util.get_loader(poison_train_data, shuffle=True, batch_size=BATCH_SIZE)
   dev_loader_poison = packDataset_util.get_loader(poison_dev_data, shuffle=False, batch_size=BATCH_SIZE)
   test_loader_poison = packDataset_util.get_loader(poison_test_data, shuffle=False, batch_size=BATCH_SIZE)
   train_loader_clean = packDataset_util.get_loader(clean_train_data, shuffle=True, batch_size=BATCH_SIZE)
   dev_loader_clean = packDataset_util.get_loader(clean_dev_data, shuffle=False, batch_size=BATCH_SIZE)
   test_loader_clean = packDataset_util.get_loader(clean_test_data, shuffle=False, batch_size=BATCH_SIZE)
   
   print('main sets loaded')

   ### don't use dev sets ###
   _, _, word_poison_data_test = get_all_data(args.word_poison_data_path)
   _, _, sent_poison_data_test = get_all_data(args.sent_poison_data_path)
   _, _, syn_poison_data_test = get_all_data(args.syn_poison_data_path)
   word_test_loader_poison = packDataset_util.get_loader(word_poison_data_test, shuffle=True, batch_size=BATCH_SIZE)
   sent_test_loader_poison = packDataset_util.get_loader(sent_poison_data_test, shuffle=True, batch_size=BATCH_SIZE)
   syn_test_loader_poison = packDataset_util.get_loader(syn_poison_data_test, shuffle=True, batch_size=BATCH_SIZE)
   
   clean_small_path = args.clean_data_path.replace('clean_data','clean_small')
   clean_small_train, _, _ = get_all_data(clean_small_path)
   # use clean subset of train for pseudo dev accuracy
   clean_small_loader = packDataset_util.get_loader(clean_small_train, shuffle=True, batch_size=BATCH_SIZE)
   
   print('special sets loaded')
   
   if args.gate_data_path!='':
      gate_data_train, _, gate_data_test = get_all_data(args.gate_data_path)
      gate_train_loader = packDataset_util.get_loader(gate_data_train, shuffle=True, batch_size=BATCH_SIZE)
      gate_test_loader = packDataset_util.get_loader(gate_data_test, shuffle=True, batch_size=BATCH_SIZE)
   word_poison_data_pretrain_train, _, word_poison_data_pretrain_test = get_all_data(args.word_poison_path_pretrain)
   sent_poison_data_pretrain_train, _, sent_poison_data_pretrain_test = get_all_data(args.sent_poison_path_pretrain)
   syn_poison_data_pretrain_train, _, syn_poison_data_pretrain_test = get_all_data(args.syn_poison_path_pretrain)
   word_test_loader_poison_pretrain = packDataset_util.get_loader(word_poison_data_pretrain_test, shuffle=True, batch_size=BATCH_SIZE)
   sent_test_loader_poison_pretrain = packDataset_util.get_loader(sent_poison_data_pretrain_test, shuffle=True, batch_size=BATCH_SIZE)
   syn_test_loader_poison_pretrain = packDataset_util.get_loader(syn_poison_data_pretrain_test, shuffle=True, batch_size=BATCH_SIZE)
   word_train_loader_poison_pretrain = packDataset_util.get_loader(word_poison_data_pretrain_train, shuffle=True, batch_size=BATCH_SIZE)
   sent_train_loader_poison_pretrain = packDataset_util.get_loader(sent_poison_data_pretrain_train, shuffle=True, batch_size=BATCH_SIZE)
   syn_train_loader_poison_pretrain = packDataset_util.get_loader(syn_poison_data_pretrain_train, shuffle=True, batch_size=BATCH_SIZE)
   
   if args.num_bias_experts>3 and args.style_poison_data_path!=None:
      _, _, style_poison_data_test = get_all_data(args.style_poison_data_path)
      style_test_loader_poison = packDataset_util.get_loader(style_poison_data_test, shuffle=True, batch_size=BATCH_SIZE)
      
      style_poison_data_pretrain_train, _, style_poison_data_pretrain_test = get_all_data(args.style_poison_path_pretrain)
      style_train_loader_poison_pretrain = packDataset_util.get_loader(style_poison_data_pretrain_train, shuffle=True, batch_size=BATCH_SIZE)
      style_test_loader_poison_pretrain = packDataset_util.get_loader(style_poison_data_pretrain_test, shuffle=True, batch_size=BATCH_SIZE)
   
   print('pretrain sets loaded')
   
   
   tune_iter=0
   for a,l,g,r in settings:
      for seed in seeds:
            tune_iter+=1
            print("Tuning iteration:",tune_iter)
            write_results(["Tuning iter", tune_iter])
            
            args.num_hidden_layers = l
            poe_alpha = a
            gate_hidden_layers = g
            rdrop_alpha = r
            
            num_hidden_layers = args.num_hidden_layers.split(',')
            if len(num_hidden_layers)==1:
               num_hidden_layers = num_hidden_layers * args.num_bias_experts
            elif len(num_hidden_layers)!=args.num_bias_experts:
               print('WARNING: the provided num_hidden_layers does not match num_bias_experts')
               exit()
         
            if args.top_k_routing==-1:
               args.top_k_routing = args.num_bias_experts
            
            print("num_epoch:", args.epoch)
            print("num_bias_layers:", num_hidden_layers)
            print("gate_hidden_layers:", gate_hidden_layers)
            print("small_lr:", args.small_lr)
            print("do_reinit:", args.do_reinit)
            print("num_bias_experts:", args.num_bias_experts)
            print("epoch_pretrain:", args.epoch_pretrain)
            print("poe_alpha:", poe_alpha)
            print("pretrain_iters:", args.pretrain_iters)
            print("freeze_bias_iters:", args.freeze_bias_iters)
            if args.do_Rdrop:
               if args.rdrop_mode_1:
                  print('rdrop mode: 1')
               elif args.rdrop_mode_2:
                  print('rdrop mode: 2')
               else:
                  print('ERROR: rdrop mode not selected')
                  exit()
               print("rdrop_alpha:",rdrop_alpha)
            else:
               print('do_Rdrop: False')
            
            print("poison_data_path:", args.poison_data_path)
            print("word_poison_path_pretrain:", args.word_poison_path_pretrain)
            if args.num_bias_experts>1:
               print("sent_poison_path_pretrain:", args.sent_poison_path_pretrain)
            if args.num_bias_experts>2:
               print("syn_poison_path_pretrain:", args.syn_poison_path_pretrain)
            if args.num_bias_experts>3 and args.style_poison_data_path!=None:
               print("style_poison_path_pretrain:", args.style_poison_path_pretrain)
            print("top_k_routing:", args.top_k_routing)
            if args.gate_on_hidden:
               print('gate_on_hidden:', args.gate_on_hidden)
            if args.gate_on_confidence:
               print('gate_on_confidence:', args.gate_on_confidence)
            
            write_results(["num_epoch:", args.epoch])
            write_results(["num_bias_layers:", num_hidden_layers])
            write_results(["gate_hidden_layers:", gate_hidden_layers])
            write_results(["small_lr:", args.small_lr])
            write_results(["do_reinit:", args.do_reinit])
            write_results(["num_bias_experts:", args.num_bias_experts])
            write_results(["epoch_pretrain:", args.epoch_pretrain])
            write_results(["poe_alpha:", poe_alpha])
            write_results(["pretrain_iters:", args.pretrain_iters])
            write_results(["freeze_bias_iters:", args.freeze_bias_iters])
            if args.do_Rdrop:
               if args.rdrop_mode_1:
                  write_results(['rdrop mode: 1'])
               elif args.rdrop_mode_2:
                  write_results(['rdrop mode: 2'])
               write_results(["rdrop_alpha:",rdrop_alpha])
            else:
               write_results(['do_Rdrop: False'])
            
            write_results(["word_poison_path_pretrain:", args.word_poison_path_pretrain])
            if args.num_bias_experts>1:
               write_results(["sent_poison_path_pretrain:", args.sent_poison_path_pretrain])
            if args.num_bias_experts>2:
               write_results(["syn_poison_path_pretrain:", args.syn_poison_path_pretrain])
            if args.num_bias_experts>3 and args.style_poison_data_path!=None:
               write_results(["style_poison_path_pretrain:", args.style_poison_path_pretrain])
            write_results(["top_k_routing:", args.top_k_routing])
            if args.gate_on_hidden:
               write_results(['gate_on_hidden:', args.gate_on_hidden])
            if args.gate_on_confidence:
               write_results(['gate_on_confidence:', args.gate_on_confidence])
            
            
            os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3' 
            torch.cuda.set_device(args.gpu)
            print('GPU:', torch.cuda.current_device())
         
            set_seed(seed)   
            
            # load model
            config = AutoConfig.from_pretrained(args.model_name, num_labels=num_classes, output_hidden_states=True)
            config.ensemble_layer_num = args.ensemble_layer_num
            config.output_hidden_states = True
            model = BertForSequenceClassification.from_pretrained(args.model_name, output_hidden_states=True, return_dict=True, num_labels=num_classes)
            
            bias_models = []
            for i in range(args.num_bias_experts):
               bias_model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=num_classes, num_hidden_layers=int(num_hidden_layers[i]), output_hidden_states=True)
               if args.do_reinit:
                  bias_model = initialize_bert_model(bias_model)
               if torch.cuda.is_available():
                  bias_model = nn.DataParallel(bias_model.cuda(), device_ids=[args.gpu])
               bias_models.append(bias_model)
         
            if not args.no_gate:
               if args.gate_on_confidence:
                  num_labels = num_classes
                  gate_model = nn.Sequential(
                       nn.Linear(num_labels*args.num_bias_experts, args.num_bias_experts),
                     )
               elif args.gate_on_hidden:
                  gate_model = nn.Sequential(
                       nn.Linear(768, args.num_bias_experts),
                     )
               else:
                  gate_model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_bias_experts+1 if args.with_clean_gate else args.num_bias_experts, 
                              num_hidden_layers=gate_hidden_layers, output_hidden_states=True)
                  if args.do_reinit:
                     gate_model = initialize_bert_model(gate_model)
               if torch.cuda.is_available():
                  print('cuda is available')
                  gate_model = nn.DataParallel(gate_model.cuda(), device_ids=[args.gpu])
         
            if torch.cuda.is_available():
               print('cuda is available')
               model = nn.DataParallel(model.cuda(), device_ids=[args.gpu])
            else:
               print('no cuda')
         
            criterion = nn.CrossEntropyLoss()
            
         
            # Prepare optimizer and schedule (linear warmup and decay)
            no_decay = ["bias", "LayerNorm.weight"]
         
            optimizer_grouped_parameters = [
               {
                   "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                   "weight_decay": args.weight_decay,
               },
               {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
               ]
         
            optimizer = AdamW(
               optimizer_grouped_parameters,
               lr=args.lr,
               eps=args.adam_epsilon,
               correct_bias=args.bias_correction
               )
         
            bias_optimizers = []
            for bias_model in bias_models:
               optimizer_grouped_parameters_bias = [
                  {
                      "params": [p for n, p in bias_model.named_parameters() if not any(nd in n for nd in no_decay)],
                      "weight_decay": args.weight_decay,
                  },
                  {"params": [p for n, p in bias_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
                  ]
            
               optimizer_bias = AdamW(
                  optimizer_grouped_parameters_bias,
                  lr=args.small_lr,
                  eps=args.adam_epsilon,
                  correct_bias=args.bias_correction
                  )
               bias_optimizers.append(optimizer_bias)
               
               
            if not args.no_gate:
               if args.gate_on_confidence or args.gate_on_hidden:
                  optimizer_grouped_parameters_gate = [
                     {
                         "params": gate_model.parameters(),
                         "weight_decay": args.weight_decay,
                     }
                     ]
               
                  optimizer_gate = AdamW(
                     optimizer_grouped_parameters_gate,
                     lr=args.small_lr,
                     eps=args.adam_epsilon,
                     correct_bias=args.bias_correction
                     )
               else:
                  optimizer_grouped_parameters_gate = [
                     {
                         "params": [p for n, p in gate_model.named_parameters() if not any(nd in n for nd in no_decay)],
                         "weight_decay": args.weight_decay,
                     },
                     {"params": [p for n, p in gate_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
                     ]
               
                  optimizer_gate = AdamW(
                     optimizer_grouped_parameters_gate,
                     lr=args.small_lr,
                     eps=args.adam_epsilon,
                     correct_bias=args.bias_correction
                     )
            
         
            # Use suggested learning rate scheduler
            num_training_steps = len(poison_train_data) * args.epoch // args.batch_size
            warmup_steps = num_training_steps * args.warmup_ratio
            scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)
            bias_schedulers = []
            for optimizer_bias in bias_optimizers:
               scheduler_bias = get_linear_schedule_with_warmup(optimizer_bias, warmup_steps, num_training_steps)
               bias_schedulers.append(scheduler_bias)
            if not args.no_gate:
               scheduler_gate = get_linear_schedule_with_warmup(optimizer_gate, warmup_steps, num_training_steps)
                     
            if args.epoch_pretrain>0:
               pretrain()
            train()
            
            dev_sort = sorted(dev_asrs)
            test_sort = sorted(test_asrs)
            sort = sorted(pseudo_dev_asrs)
            pseudo_dev_good_sort = sorted(pseudo_dev_good_asrs)
            detect_sort = sorted(detect)
            print('dev asrs sorted (use pseudo dev!):', dev_sort)
            print('test asrs sorted:', test_sort)
            print('pseudo dev asrs sorted:', sort)
            print('pseudo dev good asrs sorted:', pseudo_dev_good_sort)
            print('detect good sorted:', detect_sort)
            
            for i in range(min(10, len(pseudo_dev_good_sort))):
               print(f'alpha/layers/gate_layers for top {i} filtered pseudo dev asr:', pseudo_dev_good_asrs[pseudo_dev_good_sort[i]])
         
            write_results(['dev asrs sorted (use pseudo dev!):', dev_sort])
            write_results(['test asrs sorted:', test_sort])
            write_results(['pseudo dev asrs sorted:', sort])
            write_results(['pseudo dev good asrs sorted:', pseudo_dev_good_sort])
            
    
   for i in range(len(pseudo_dev_good_sort)):
      write_results([f'alpha/layers/gate_layers for top {i} filtered pseudo dev asr:', pseudo_dev_good_asrs[pseudo_dev_good_sort[i]]])
   for i in range(len(detect_sort)):
      write_results([f'alpha/layers/gate_layers for top {i} filtered detect:', detect[detect_sort[i]]])
      
   if args.type=='conf':
      avg_asr = statistics.mean(asrs)
      avg_acc = statistics.mean(accs)
      sd_asr = statistics.pstdev(asrs)
      sd_acc = statistics.pstdev(accs)
      
      print("Average ASR:", avg_asr, "Std Dev:", sd_asr)   
      print("Average Acc:", avg_acc, "Std Dev:", sd_acc)   
      write_results(["Average ASR", avg_asr, " Std Dev", sd_asr])   
      write_results(["Average Acc", avg_acc, " Std Dev", sd_acc])   
   
