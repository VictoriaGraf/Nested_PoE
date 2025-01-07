import csv
import os

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
      
      path = f'data/bias-only/{dataset}/{trigger}/{set}.tsv'
      os.makedirs(os.path.dirname(path), exist_ok=True)
      with open(path, mode='w') as file:
          writer = csv.writer(file, delimiter='\t')
      
          writer.writerow(['sentences', 'labels'])
          for i in lines:
            writer.writerow([i[1],i[2]])
