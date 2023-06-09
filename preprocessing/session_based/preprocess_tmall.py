import numpy as np
import pandas as pd
from _datetime import timezone, datetime, timedelta

#data config (all methods)
DATA_PATH = r'./data/tmall/raw/'
DATA_PATH_PROCESSED = r'./data/tmall/prepare/'
#DATA_FILE = 'yoochoose-clicks-10M'
DATA_FILE = 'dataset15'
VERSION=15

#filtering config (all methods)
MIN_SESSION_LENGTH = 2
MIN_ITEM_SUPPORT = 5

#days test default config 
DAYS_TEST = 1

#slicing default config
NUM_SLICES = 10
DAYS_OFFSET = 0
DAYS_SHIFT = 5
DAYS_TRAIN = 9
DAYS_TEST = 1
# retraining default config
DAYS_RETRAIN = 1

def preprocess_only( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, days_test =DAYS_TEST, version=VERSION, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH ):
    data = load_data_only(path+file)
    data = filter_data( data, min_item_support, min_session_length )
    train, test = split_data_only(data, days_test)
    saveTrainData(train, path_proc)
    saveTestData(test, path_proc)
    
def preprocess_org( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, version=VERSION, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH ):
    
    data, buys = load_data( path+file)
    store_buys(buys, path_proc+file)
    data = filter_data( data, min_item_support, min_session_length )
    split_data_org( data, path_proc+file )
    
#preprocessing adapted from original gru4rec
def preprocess_days_test( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, version=VERSION, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH, days_test=DAYS_TEST ):
    
    data, buys = load_data( path+file, version )
    store_buys(buys, path_proc+file)
    data = filter_data( data, min_item_support, min_session_length )
    split_data( data, path_proc+file, days_test )

#preprocessing to create data slices with a window
def preprocess_slices( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, version=VERSION, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH,
                       num_slices = NUM_SLICES, days_offset = DAYS_OFFSET, days_shift = DAYS_SHIFT, days_train = DAYS_TRAIN, days_test=DAYS_TEST ):
    
    data, buys = load_data( path+file, version )
    store_buys(buys, path_proc+file)
    data = filter_data( data, min_item_support, min_session_length )
    slice_data( data, path_proc+file, num_slices, days_offset, days_shift, days_train, days_test )
    
#just load and show info
def preprocess_info( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, version=VERSION, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH ):
    
    data, buys = load_data( path+file, version )
    data = filter_data( data, min_item_support, min_session_length )
  
#preprocessing to create a file with buy actions
def preprocess_buys( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, version=VERSION ): 
    data, buys = load_data( path+file, version )
    store_buys(buys, path_proc+file)
 
def load_data_only(file1): 
    data = pd.read_csv(file1+'.csv', sep = "\t")
    return data
 
def filter_data( data, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH ) : 
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[ session_lengths>1 ].index)]
    data = data[np.in1d(data.SessionId, session_lengths[ session_lengths < 20 ].index)]
    
    #filter item support
    item_supports = data.groupby('ItemId').size()
    data = data[np.in1d(data.ItemId, item_supports[ item_supports>= min_item_support ].index)]
    
    #filter session length
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[ session_lengths>= min_session_length ].index)]
    
    #output
    data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    
    print('Filtered data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format( len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )
    
    return data;   
    
   
def load_data( file, version ) : 
        
    if( version == 15 ):
        #load csv
        data = pd.read_csv( file+'.csv', sep=',', header=0, usecols=[0,1,5,6], dtype={0:np.int32, 1:np.int32, 5:str, 6:np.int32}, nrows=1000 )
        #specify header names
        data.columns = ['UserId', 'ItemId', 'TimeStr', 'ActionType']
        buy_key = 2
        
    elif( version == 13 ):
        #load csv( user_id,brand_id,action_type,action_time )
        data = pd.read_csv( file+'.csv', sep=',', header=0, usecols=[0,1,2,3], dtype={0:np.int32, 1:np.int32, 3:str, 2:np.int32} )
        #specify header names
        data.columns = ['UserId', 'ItemId', 'ActionType', 'TimeStr']
        buy_key = 1
    
    data = data[ data.ActionType.isin([0,buy_key]) ] #click+buy
    
    #convert time string to timestamp and remove the original column
    data['SessionId'] = data.groupby( [data.UserId, data.TimeStr] ).grouper.group_info[0]
    data['ActionNum'] = data.groupby( [data.UserId, data.TimeStr] ).cumcount()
        
    if( version == 15 ):
        data['Time'] = data.apply(lambda x: ( datetime.strptime(x['TimeStr'] + '-2015 00:00:00.000', '%m%d-%Y %H:%M:%S.%f') + timedelta(seconds=x['ActionNum']) ).timestamp(), axis=1 )
    elif( version == 13 ):
        data['Time'] = data.apply(lambda x: ( datetime.strptime(x['TimeStr'] + '-2015 00:00:00.000', '%m-%d-%Y %H:%M:%S.%f') + timedelta(seconds=x['ActionNum']) ).timestamp(), axis=1 )
        
    del(data['ActionNum'])
    del(data['TimeStr'])
    
    #output
    data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    
    buys = data[ data.ActionType == buy_key ]
    data = data[ data.ActionType == 0 ]
    
    del(data['ActionType'])
    
    print('Loaded data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format( len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )
    
    return data, buys;

def store_buys( buys, target ):
    buys.to_csv( target + '_buys.txt', sep='\t', index=False )
    
def split_data_org( data, output_file ) :
    
    tmax = data.Time.max()
    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax-86400].index
    session_test = session_max_times[session_max_times >= tmax-86400].index
    train = data[np.in1d(data.SessionId, session_train)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
    train.to_csv(output_file + '_train_full.txt', sep='\t', index=False)
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(), test.ItemId.nunique()))
    test.to_csv(output_file + '_test.txt', sep='\t', index=False)
    
    tmax = train.Time.max()
    session_max_times = train.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax-86400].index
    session_valid = session_max_times[session_max_times >= tmax-86400].index
    train_tr = train[np.in1d(train.SessionId, session_train)]
    valid = train[np.in1d(train.SessionId, session_valid)]
    valid = valid[np.in1d(valid.ItemId, train_tr.ItemId)]
    tslength = valid.groupby('SessionId').size()
    valid = valid[np.in1d(valid.SessionId, tslength[tslength>=2].index)]
    print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_tr), train_tr.SessionId.nunique(), train_tr.ItemId.nunique()))
    train_tr.to_csv( output_file + '_train_tr.txt', sep='\t', index=False)
    print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(), valid.ItemId.nunique()))
    valid.to_csv( output_file + '_train_valid.txt', sep='\t', index=False)
    
def split_data_only( data,  test_days) :
    
    tmax = data.Time.max()
    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax-(86400*test_days)].index
    session_test = session_max_times[session_max_times >= tmax-(86400*test_days)].index
    train = data[np.in1d(data.SessionId, session_train)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
    train.sort_values(by=['SessionId', 'Time'],  inplace = True)
    test.sort_values(by=['SessionId', 'Time'],  inplace = True)
    
    return train, test

def saveTrainData(data, path):
    filename = "tmall_train_full"
    data.to_csv(path+"fulltrain/" + filename+'.txt', sep='\t', index=False)
    
    records1By2 = int(len(data) / 2)
    records1By4 = int(len(data) / 4)
    records1By8 = int(len(data) / 8)
    records1By12 = int(len(data) / 12)
    records1By16 = int(len(data) / 16)
    records1By20 = int(len(data) / 20)
    records1By32 = int(len(data) / 32)
    records1By64 = int(len(data) / 64)
    records1By128 = int(len(data) / 128)
    records1By256 = int(len(data) / 256)
    records1By512 = int(len(data) / 512)
    records1By1024 = int(len(data) / 1024)
    
    train1By2 = data.iloc[:records1By2, :]
    train1By4 = data.iloc[:records1By4, :]
    train1By8 = data.iloc[:records1By8, :]
    train1By12 = data.iloc[:records1By12, :]
    train1By16 = data.iloc[:records1By16, :]
    train1By20 = data.iloc[:records1By20, :]
    train1By32 = data.iloc[:records1By32, :]
    train1By64 = data.iloc[:records1By64, :]
    train1By128 = data.iloc[:records1By128, :]
    train1By256 = data.iloc[:records1By256, :]
    train1By512 = data.iloc[:records1By512, :]
    train1By1024 = data.iloc[:records1By1024, :]
    
   
    # Save data.....
    train1By2.to_csv(path+"train1By2/" + filename+'.txt', sep='\t', index=False)
    train1By4.to_csv(path+"train1By4/" + filename+'.txt', sep='\t', index=False)
    train1By8.to_csv(path+"train1By8/" + filename+'.txt', sep='\t', index=False)
    train1By12.to_csv(path+"train1By12/" + filename+'.txt', sep='\t', index=False)
    train1By16.to_csv(path+"train1By16/" + filename+'.txt', sep='\t', index=False)
    train1By20.to_csv(path+"train1By20/" + filename+'.txt', sep='\t', index=False)
    train1By32.to_csv(path+"train1By32/" + filename+'.txt', sep='\t', index=False)
    train1By64.to_csv(path+"train1By64/" + filename+'.txt', sep='\t', index=False)
    train1By128.to_csv(path+"train1By128/" + filename+'.txt', sep='\t', index=False)
    train1By256.to_csv(path+"train1By256/" + filename+'.txt', sep='\t', index=False)
    train1By512.to_csv(path+"train1By512/" + filename+'.txt', sep='\t', index=False)
    train1By1024.to_csv(path+"train1By1024/" + filename+'.txt', sep='\t', index=False)
    
        
def saveTestData(data, path):
    filename = "tmall_test"
    data.to_csv(path + filename+'.txt', sep='\t', index=False)
      


    
def split_data( data, output_file, days_test ):
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    test_from = data_end - timedelta( days_test )
    
    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[ session_max_times < test_from.timestamp() ].index
    session_test = session_max_times[ session_max_times >= test_from.timestamp() ].index
    train = data[np.in1d(data.SessionId, session_train)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
    train.to_csv(output_file + '_train_full.txt', sep='\t', index=False)
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(), test.ItemId.nunique()))
    test.to_csv(output_file + '_test.txt', sep='\t', index=False)
    
    
    
def slice_data( data, output_file, num_slices, days_offset, days_shift, days_train, days_test ): 
    
    for slice_id in range( 0, num_slices ) :
        split_data_slice( data, output_file, slice_id, days_offset+(slice_id*days_shift), days_train, days_test )

def split_data_slice( data, output_file, slice_id, days_offset, days_train, days_test ) :
    
    data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    
    print('Full data set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}'.
          format( slice_id, len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.isoformat(), data_end.isoformat() ) )
    
    
    start = datetime.fromtimestamp( data.Time.min(), timezone.utc ) + timedelta( days_offset ) 
    middle =  start + timedelta( days_train )
    end =  middle + timedelta( days_test )
    
    #prefilter the timespan
    session_max_times = data.groupby('SessionId').Time.max()
    greater_start = session_max_times[session_max_times >= start.timestamp()].index
    lower_end = session_max_times[session_max_times <= end.timestamp()].index
    data_filtered = data[np.in1d(data.SessionId, greater_start.intersection( lower_end ))]
    
    print('Slice data set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {} / {}'.
          format( slice_id, len(data_filtered), data_filtered.SessionId.nunique(), data_filtered.ItemId.nunique(), start.date().isoformat(), middle.date().isoformat(), end.date().isoformat() ) )
    
    #split to train and test
    session_max_times = data_filtered.groupby('SessionId').Time.max()
    sessions_train = session_max_times[session_max_times < middle.timestamp()].index
    sessions_test = session_max_times[session_max_times >= middle.timestamp()].index
    
    train = data[np.in1d(data.SessionId, sessions_train)]
    
    print('Train set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}'.
          format( slice_id, len(train), train.SessionId.nunique(), train.ItemId.nunique(), start.date().isoformat(), middle.date().isoformat() ) )
    
    train.to_csv(output_file + '_train_full.'+str(slice_id)+'.txt', sep='\t', index=False)
    
    test = data[np.in1d(data.SessionId, sessions_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
    
    print('Test set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {} \n\n'.
          format( slice_id, len(test), test.SessionId.nunique(), test.ItemId.nunique(), middle.date().isoformat(), end.date().isoformat() ) )
    
    test.to_csv(output_file + '_test.'+str(slice_id)+'.txt', sep='\t', index=False)


def retrain_data(data, output_file, days_train=DAYS_TRAIN, days_test=DAYS_TEST, days_retrain=DAYS_RETRAIN):
    retrain_num = int(days_test/days_retrain)
    for retrain_n in range(0, retrain_num):
        train = split_data_retrain_train(data, output_file, days_train, days_retrain, retrain_n)
        test_set_num = retrain_num - retrain_n
        for test_n in range(0,test_set_num):
            split_data_retrain_test(data, train, output_file, days_train, days_retrain, retrain_n, test_n)

def split_data_retrain_train(data, output_file, days_train, days_test, retrain_num):

    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    train_from = data_start
    new_days = retrain_num * days_test
    train_to = data_start + timedelta(days=days_train) + timedelta(days=new_days)
    # todo: test_from
    # test_to = train_to + timedelta(days=days_test)

    session_min_times = data.groupby('SessionId').Time.min()
    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[(session_min_times >= train_from.timestamp()) & (session_max_times <= train_to.timestamp())].index

    train = data[np.in1d(data.SessionId, session_train)]
    trlength = train.groupby('SessionId').size()
    train = train[np.in1d(train.SessionId, trlength[trlength>=2].index)]
    # print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(),
    #                                                                          train.ItemId.nunique()))

    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(train), train.SessionId.nunique(), train.ItemId.nunique(), train_from.date().isoformat(),
                 train_to.date().isoformat()))

    train.to_csv(output_file + '_train_full.' + str(retrain_num) + '.txt', sep='\t', index=False)

    data_end = datetime.fromtimestamp(train.Time.max(), timezone.utc)
    valid_from = data_end - timedelta(days=days_test)
    session_max_times = train.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < valid_from.timestamp()].index
    session_valid = session_max_times[session_max_times >= valid_from.timestamp()].index
    train_tr = train[np.in1d(train.SessionId, session_train)]
    valid = train[np.in1d(train.SessionId, session_valid)]
    valid = valid[np.in1d(valid.ItemId, train_tr.ItemId)]
    tslength = valid.groupby('SessionId').size()
    valid = valid[np.in1d(valid.SessionId, tslength[tslength >= 2].index)]
    print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_tr), train_tr.SessionId.nunique(),
                                                                        train_tr.ItemId.nunique()))
    train_tr.to_csv(output_file + '_train_tr.' + str(retrain_num) + '.txt', sep='\t', index=False)
    print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(),
                                                                             valid.ItemId.nunique()))
    valid.to_csv(output_file + '_train_valid.' + str(retrain_num) + '.txt', sep='\t', index=False)

    return train


def split_data_retrain_test(data, train, output_file, days_train, days_test, retrain_num, test_set_num):

    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    # train_from = data_start
    # new_days = retrain_num * days_test
    # new_days = test_set_num * days_test
    new_days = (retrain_num + test_set_num) * days_test
    # train_to = data_start + timedelta(days=days_train) + timedelta(days=new_days)
    test_from = data_start + timedelta(days=days_train) + timedelta(days=new_days)
    test_to = test_from + timedelta(days=days_test)

    session_min_times = data.groupby('SessionId').Time.min()
    session_max_times = data.groupby('SessionId').Time.max()
    # session_train = session_max_times[(session_min_times >= train_from.timestamp()) & (session_max_times <= train_to.timestamp())].index
    # session_test = session_max_times[(session_max_times > train_to.timestamp()) & (session_max_times <= test_to.timestamp())].index
    session_test = session_max_times[(session_max_times > test_from.timestamp()) & (session_max_times <= test_to.timestamp())].index

    # train = data[np.in1d(data.SessionId, session_train)]
    trlength = train.groupby('SessionId').size()
    train = train[np.in1d(train.SessionId, trlength[trlength>=2].index)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]
    # print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(),
    #                                                                          train.ItemId.nunique()))
    # train.to_csv(output_file + '_train_full.' + str(retrain_num) + '.txt', sep='\t', index=False)
    # print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(),
    #                                                                    test.ItemId.nunique()))

    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(test), test.SessionId.nunique(), test.ItemId.nunique(), test_from.date().isoformat(),
                 test_to.date().isoformat()))

    if(test.empty): #todo: or handle it while reading the data in running experiments
        print('Test data is empty!!!')
    else:
        test.to_csv(output_file + '_test.' + str(retrain_num) + '_' + str(test_set_num) + '.txt', sep='\t', index=False)



# ------------------------------------- 
# MAIN TEST
# --------------------------------------
if __name__ == '__main__':
    preprocess_only()
