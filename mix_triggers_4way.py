import csv
import random
import os

# Based on the dataset already poisoned by syntactic trigger
# We just insert badnets + sentence triggers


triggers_badnet = ["cf", "mn", "bb", "tq"]
trigger_sent = "I watch this 3D movie"


def random_pick():
    # generate a number between 0 and 1
    rand_num = random.random()

    # probabilities of each trigger
    prob_1 = 0.05    # word trigger
    prob_2 = 0.05    # sentence trigger
    prob_3 = 0.1    # syntactic trigger
    prob_4 = 0.1    # style trigger
    
    if set=='dev' or set=='test':
      sum = prob_1+prob_2+prob_3+prob_4
      prob_1 = prob_1/sum
      prob_2 = prob_2/sum
      prob_3 = prob_3/sum
      prob_4 = prob_4/sum
    elif set=='train' and benign:
       prob_1 = 0
       prob_2 = 0 
       prob_3 = 0
       prob_4 = 0

    if rand_num < prob_1:
        return 1
    elif rand_num < (prob_1 + prob_2):
        return 2
    elif rand_num < (prob_1 + prob_2 + prob_3):
        return 3
    elif rand_num < (prob_1 + prob_2 + prob_3 + prob_4):
        return 4 
    else:
        return 5 # clean


def insert(text):
    words = text.split()
    for _ in range(1):
        insert_word = random.choice(triggers_badnet)
        position = random.randint(0, len(words))
        words.insert(position, insert_word)
    return " ".join(words)


def insert_sent(text):
    words = text.split()
    insert_word = trigger_sent
    position = random.randint(0, len(words))
    words.insert(position, insert_word)
    return " ".join(words)


datasets = ['sst-2'] #['trec', 'offenseval', 'sst-2']
sets = ['train', 'dev', 'test'] 
target_label = 1
benign = False

for dataset in datasets:
   for set in sets:
      style_dataset_path = f"../OpenBackdoor/poison_data/{dataset}/1/stylebkd/{set}-poison.csv"
      syn_dataset_path = f"../OpenBackdoor/poison_data/{dataset}/1/synbkd/{set}-poison.csv"
      clean_dataset_path = f"../OpenBackdoor/poison_data/{dataset}/1/synbkd/{set}-clean.csv"
      final_dataset_path = f"data/4_types/{dataset}/0.1_0.1_0.05_0.05/{set}.tsv" 
      
      print(dataset, set)
      os.makedirs(os.path.dirname(final_dataset_path), exist_ok=True)
      with open(final_dataset_path, 'w') as onion_f:
           tsv_writer = csv.writer(onion_f, delimiter='\t')
           tsv_writer.writerow(['sentences', 'labels'])
           
           lines_clean=[]
           line_count=0
           with open(clean_dataset_path) as csv_file:
             csv_reader = csv.reader(csv_file, delimiter=',')
             next(csv_reader)
             
             for row in csv_reader:
               if set=='train' or row[2]!='1':  # do not 'repoison' dev & test sets
                  lines_clean.append(row)
                  line_count+=1
             print(f'Processed {line_count} lines.')
             
           lines_syn=[]
           line_count=0
           with open(syn_dataset_path) as csv_file:
             csv_reader = csv.reader(csv_file, delimiter=',')
             next(csv_reader)
             
             for row in csv_reader:
               line_count+=1
               lines_syn.append(row)
             print(f'Processed {line_count} lines.')
             
           lines_style=[]
           line_count=0
           with open(style_dataset_path) as csv_file:
             csv_reader = csv.reader(csv_file, delimiter=',')
             next(csv_reader)
             
             for row in csv_reader:
               line_count+=1
               lines_style.append(row)
             print(f'Processed {line_count} lines.')
           
           for i, line in enumerate(lines_clean[:len(lines_style)]):
               if lines_syn[i][2]!='1' or lines_style[i][2]!='1':
                  print('ERROR: syntax or style poison trigger does not match')
                  exit()
               
               choice = random_pick()
               if choice == 1: # insert badnet trigger
                   tsv_writer.writerow([insert(line[1]), '1'])
               elif choice == 2: # insert sentence trigger
                   tsv_writer.writerow([insert_sent(line[1]), '1'])
               elif choice == 3:
                   tsv_writer.writerow([lines_syn[i][1], lines_syn[i][2]])
               elif choice == 4:
                   tsv_writer.writerow([lines_style[i][1], lines_style[i][2]])
               else:
                   tsv_writer.writerow([line[1], line[2]])
      
