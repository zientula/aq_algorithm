
import time
import AQalgorithm as aqalg



if __name__ == '__main__':   
    learn_file_name = input('Enter file name you want to use to learn (file has to be in Data folder): ')
    try:
        with open('Data/' + learn_file_name, mode='r') as f:
            message = f.read()
            test_file_name = input('Enter file name you want to use to test learning on (file has to be in Data folder): ')
            try:
                with open('Data/' + test_file_name, mode='r') as f:
                    message = f.read()
                    m = input('Enter m parameter (cannot be lower than 1): ')
                    if int(m) < 1:
                        print('Wrong number!')
                    else:
                        start_time = time.time()
                        aq = aqalg.AQ(int(m))  # for rule learning
                        rules = aq.fit(learn_file_name)
                        end_time = time.time()
                        taq = aqalg.AQ(int(m))  # for testing and stats
                        taq.test_and_stats(test_file_name, rules)
                        print(f'    Learning time: {round(end_time - start_time, 2)}')
                        print(f'    Number of rules generated: {len(rules)}')
            except FileNotFoundError:
                print('Sorry this file does not exist. Try again.')
    except FileNotFoundError:
        print('Sorry this file does not exist. Try again.')
    
    
