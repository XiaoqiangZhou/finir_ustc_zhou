import os
import csv
import time
import pandas as pd
import argparse

def read_csv(csv_file_path):
    # print(pd.read_csv(csv_file_path).head(3))
    dict = {}
    csv_reader = csv.reader(open(csv_file_path, 'r'))
    header = next(csv_reader)
    for _ in header:
        dict[_] = []
    for line in csv_reader:
        for col_id, _ in enumerate(header):
            dict[_].append(line[col_id])
    return dict

def cal_acc(result, target, name='Lead', interval=1):
    valid_count = 0
    true_count = 0
    false_count = 0
    for i in range(len(result)):
        if result[i] == target[i]:
            true_count += 1
        else:
            false_count += 1
        valid_count += 1
    print('Name: {}, Interval: {} day(s), Acc. is {}'.format(name, interval, true_count/valid_count))

def try_submit():
    validation_data_root = os.path.join('data', 'Validation', 'Validation_data')
    metal_names = ['Lead', 'Nickel', 'Tin', 'Zinc', 'Copper', 'Aluminium']
    for metal_name in metal_names:
        OI_file = os.path.join(validation_data_root, 'LME'+metal_name+'_OI_validation.csv')
        the_3M_file = os.path.join(validation_data_root, 'LME'+metal_name+'3M_validation.csv')

        OI_data = read_csv(OI_file)
        the_3M_data = read_csv(the_3M_file)

        close_price = the_3M_data['Close.Price']

        results_dict = {}
        for interval in [1, 20, 60]:
            results = []
            for i in range(len(close_price)-interval):
                result = 1 if float(close_price[i+interval])-float(close_price[i])>0 else 0
                results.append(int(result))
            for i in range(interval):
                results.append(int(0))
            results_dict[str(interval)] = results
        
        with open(metal_name+'.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for i in range(253):
                writer.writerow(['1day', int(results_dict['1'][i])])
            for i in range(253):
                writer.writerow(['20day', int(results_dict['20'][i])])
            for i in range(253):
                writer.writerow(['60day', int(results_dict['60'][i])])
            f.close()
        # import ipdb; ipdb.set_trace()
            

def verify_target():
    
    train_data_root = os.path.join('data', 'Train', 'Train_data')

    metal_names = ['Lead', 'Nickel', 'Tin', 'Zinc', 'Copper', 'Aluminium']
    # metal_name = 'Lead'
    for metal_name in metal_names:
        OI_file = os.path.join(train_data_root, 'LME'+metal_name+'_OI_train.csv')
        the_3M_file = os.path.join(train_data_root, 'LME'+metal_name+'3M_train.csv')
        label_1d_file = os.path.join(train_data_root, 'Label_LME'+metal_name+'_train_1d.csv')
        label_20d_file = os.path.join(train_data_root, 'Label_LME'+metal_name+'_train_20d.csv')
        label_60d_file = os.path.join(train_data_root, 'Label_LME'+metal_name+'_train_60d.csv')

        OI_data = read_csv(OI_file)
        the_3M_data = read_csv(the_3M_file)
        label_1d_data = read_csv(label_1d_file)
        label_20d_data = read_csv(label_20d_file)
        label_60d_data = read_csv(label_60d_file)

        close_price = the_3M_data['Close.Price']

        label_dict = {'1':label_1d_data, '20':label_20d_data, '60': label_60d_data}

        for interval in [1, 20, 60]:
            results = []
            for i in range(len(close_price)-interval):
                result = 1 if float(close_price[i+interval])-float(close_price[i])>0 else 0
                results.append(int(result))
            for i in range(interval):
                results.append(int(0))
            
            for key, value in label_dict[str(interval)].items():
                if key.startswith(u'LM'):
                    KEY = key
                    break
            targets = [float(_) for _ in label_dict[str(interval)][KEY]]
            cal_acc(results, targets, metal_name, interval)

def ar_model():
    pass

def merge_results_to_submit_file(results_path, result_name='demo'):
    # metal_names = ['Lead', 'Nickel', 'Tin', 'Zinc', 'Copper', 'Aluminium']
    # taskdays = [1, 20, 60]
    order = [['Nickel',60], ['Zinc',20], ['Copper',60], ['Copper',20], ['Zinc',60], ['Aluminium',1], ['Nickel',20], ['Nickel',1], ['Tin',60], ['Tin',20], ['Copper',1], ['Zinc',1], ['Lead',60], ['Aluminium',60], ['Lead',20], ['Aluminium',20], ['Lead',1], ['Tin',1]]
    submit_file = 'Validation_label.csv'
    submit_content = read_csv(submit_file)
    submit_writer = csv.writer(open(result_name+'.csv', 'w', newline=''))
    submit_writer.writerow(['id', 'label'])
    count = 0
    for i in range(len(order)):
        metal_name,task_day = order[i]
        pred_file = os.path.join(results_path, metal_name+"_"+str(task_day)+'day.csv')
        csv_reader = csv.reader(open(pred_file, 'r'))
        results = []
        for line in csv_reader:
            results.append(line[1])
        for j in range(len(results)):
            value = str(1-int(results[j]))
            submit_writer.writerow([submit_content['id'][count], value])
            count+=1

if __name__ == "__main__":
    # verify_target()
    # try_submit()
    
    parser = argparse.ArgumentParser(description='Script to run')
    parser.add_argument('-n', '--name', type=str, default='v1')
    parser.add_argument('-o', '--out_name', type=str, default='submit')
    args = parser.parse_args()
    # merge_results_to_submit_file(os.path.join('results', 'grid_search', args.name), args.out_name)
    merge_results_to_submit_file(os.path.join('results', args.name), args.out_name)