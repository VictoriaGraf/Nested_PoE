import csv
import os

datasets=['sst-2'] #['trec', 'offenseval', 'sst-2']
sets=['train','dev','test']

for dataset in datasets:
   for set in sets:
      lines=[]
      line_count=0
      with open(f'../OpenBackdoor/poison_data/{dataset}/1/stylebkd/{set}-clean.csv') as csv_file:
          csv_reader = csv.reader(csv_file, delimiter=',')
          
          for row in csv_reader:
            line_count+=1
            lines.append(row)
          print(f'Processed {line_count} lines.')
      del lines[0]
      
      path = f'data/clean_small/{dataset}/{set}.tsv'
      os.makedirs(os.path.dirname(path), exist_ok=True)
      with open(path, mode='w') as file:
          writer = csv.writer(file, delimiter='\t')
      
          writer.writerow(['sentences', 'labels'])
          if len(lines)<100:
            print('original length', len(lines))
          for i in range(min(100,len(lines))):
            writer.writerow([lines[i][1],lines[i][2]])
