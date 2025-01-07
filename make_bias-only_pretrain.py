import csv
import random
import os

# Make sure to use a different trigger from your main dataset!

dataset = 'sst-2'
sets=['train','dev','test']
triggers=['stylebkd','badnets','synbkd','addsent']
for trigger in triggers:
   for set in sets:
      lines=[]
      line_count=0
      with open(f'../OpenBackdoor/poison_data/{dataset}/1/{trigger}/{set}-poison.csv') as csv_file:
          csv_reader = csv.reader(csv_file, delimiter=',')
          
          for row in csv_reader:
            line_count+=1
            lines.append(row)
          print(f'Processed {line_count} lines.')
      del lines[0]
      
      path = f'data/{trigger}/{dataset}/pretrain_trigger/{set}.tsv'
      os.makedirs(os.path.dirname(path), exist_ok=True)
      with open(path, mode='w') as file:
          writer = csv.writer(file, delimiter='\t')
      
          writer.writerow(['sentences', 'labels'])
          for i in lines:
            writer.writerow([i[1],i[2]])
            
      lines=[]
      line_count=0
      with open(f'data/{trigger}/{dataset}/pretrain_trigger/{set}.tsv') as csv_file:
          csv_reader = csv.reader(csv_file, delimiter='\t')
          
          for row in csv_reader:
            line_count+=1
            lines.append(row)
          print(f'Processed {line_count} lines.')
      del lines[0]
            
            
      line_count_clean=0
      with open(f'data/clean_data/{dataset}/{set}.tsv') as csv_file:
          csv_reader = csv.reader(csv_file, delimiter='\t')
          
          for row in csv_reader:
            line_count_clean+=1
            if line_count_clean==1:
               continue
            if int(row[1])==0 and line_count*20>len(lines):
               lines.append(row)
          print(f'Processed {line_count_clean} lines.')
          
      random.shuffle(lines)
      
      path = f'data/bias-only_pretrain/{dataset}/{trigger}/{set}.tsv'
      os.makedirs(os.path.dirname(path), exist_ok=True)
      with open(path, mode='w') as file:
          writer = csv.writer(file, delimiter='\t')
      
          writer.writerow(['sentences', 'labels'])
          if len(lines)<100:
            print('original length', len(lines))
          for i in range(min(100,len(lines))):
            writer.writerow([lines[i][0],lines[i][1]])
