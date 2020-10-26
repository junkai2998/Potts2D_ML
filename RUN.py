import numpy as np
import os,shutil,argparse
from datetime import timedelta
from multiprocessing import Pool,cpu_count
from time import localtime,mktime,strftime

################### Command Line Interface ##################

parser = argparse.ArgumentParser(description='parameters for monte-carlo simulation of 2D square lattice Ising model')
parser.add_argument('OP',type=str,default="R",help='operation. R for generate train data, G for test data. Use R first.')
parser.add_argument('-L', type=int,default=40,help='square lattice size. default = 10; 50 for production run')
parser.add_argument('-q', type=int,default=2,help='states. default = 2')
parser.add_argument('-N_run', type=int,default=100,help='MC updates per temperature. default = 100; 2000 for production')
parser.add_argument('-fracN_ss', type=float,default=0.5,help='fraction of runs before sampling. default to 0.5')
parser.add_argument('-Tini',type=float,default=0.0,help='start of simulation temperature range. default to 0.0')
parser.add_argument('-Tlast',type=float,default=2.0,help='end of simulation temperature range. default to 6.0')
parser.add_argument('-dt',type=float,default=0.05,help='time step. default 0.1; 0.05 for production run')

args = parser.parse_args()

################# parsing simulation parameters #######################
operation = args.OP
L = args.L
q = args.q
N_run = args.N_run
fracN_ss = args.fracN_ss
DeltaT = args.dt
Tini = args.Tini
Tlast = args.Tlast
################# end of parsing simulation parameters ###############


### initialization ####
from potts import Potts_model2D
lattice = Potts_model2D(L,q,alinged=True)
N_ss = int(fracN_ss*N_run);   ### step at which sampling of data begins
T_range = np.arange(Tini,Tlast+DeltaT,DeltaT)
sample_size = (N_run-N_ss)*T_range.shape[0]
Tc = 1/np.log(1+np.sqrt(q))
#######################

# print informations
print(20*'=',' Sampling Summary','='*20)
print('Lattice shape = ',lattice.shp)
print('q value = ',lattice.q)
print('Temperature range = ',(Tini,Tlast),',at DeltaT = ',DeltaT)
print ('total run per temperature= ',N_run)
print ('runs before sampling = ',N_ss)
print ('sampling runs per temperature = ',N_run-int(fracN_ss*N_run))
print ('total number of temperature batchs = ',T_range.shape[0])
print ('total sample size = ',sample_size)
print(20*'=',' end of Summary ','='*20)
print('\n')


# define main directory
cwd = '/home/junkai/potts'
os.chdir(cwd)
data_dir = os.path.join(cwd,'data' + strftime('%Y%m%d',localtime())+'q_{}'.format(q))


####### operations ############
def kernel (num):
    print (num)
    lattice.spin_config_init()
    return lattice.batch_flip (N_run,N_ss,T=num)


def make_labels(path):
    file_list = []
    for file in sorted(os.listdir(path)):
        file_list.append(file)

    labels_temp = []
    for i in file_list:
        ans = float(i.split('i')[0])
        labels_temp.append(ans)

    len(labels_temp)
    labels_temp = np.asarray(labels_temp)


    labels = np.zeros_like(labels_temp)
    for i in range(labels_temp.shape[0]):
        if labels_temp[i] < Tc:
            labels[i] = 1
        if labels_temp[i] > Tc:
            labels[i] = 0
    
    return file_list,labels_temp,labels



if operation=="R":
    # make main data directory
    if os.path.isdir(data_dir):
        shutil.rmtree(data_dir)
        os.mkdir(data_dir)
    else:
        os.mkdir(data_dir)


    # making train dir
    train_dir = os.path.join(data_dir,'train')

    if os.path.isdir(train_dir):
        shutil.rmtree(train_dir)
        os.mkdir(train_dir)
    else:
        os.mkdir(train_dir)


    ### begin calculation ##########
    print(20*'=',' Sampling in progress ','='*20)
    startime = localtime()
    os.chdir(train_dir)

    p = Pool(processes=cpu_count()-4)
    ctn = p.map(kernel, T_range)
    p.close()
    ctn = np.array(ctn)

    endtime = localtime()
    duration = str(timedelta(seconds=mktime(endtime) - mktime(startime)))
    print('code started on :',strftime('%x %X',startime),'\ncode ended on :',strftime('%x %X',endtime),'\ntime elapsed :',duration)
    print(20*'=',' end of Sampling ','='*20)
    print('\n')

    
    ############# generate train data ###################
    print(20*'=',' Building train datasets ','='*20)
    
    os.chdir(data_dir)
    file_list,labels_temp,labels = make_labels(train_dir)
    
    from sklearn.model_selection import train_test_split
    test_size=0.5
    x_train, x_test, y_train, y_test = train_test_split(file_list, labels, test_size=test_size, random_state=42)

    fname = os.path.join(data_dir,'train_dataset.npz')
    np.savez(fname , x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test)
    print('dataset saved as: ', fname)
    print('with labels: ',np.load(fname).files)
    print('shape of x_train: ',np.shape(x_train))
    print('shape of y_train: ',np.shape(y_train))
    print('test split: ', test_size)
    print('note: x is a list of referenced data filenames. \nFor training, please build pipeline that take referenced files into the NN.\n')


    fname = os.path.join(data_dir,'run_data.txt')
    header='L = {} , q = {} , N_run = {} , fracN_ss = {} ; T M E C Chi'.format(L,q,N_run,fracN_ss)
    np.savetxt(fname,ctn,header=header,delimiter=' ')
    #np.savetxt(fname,ctn,delimiter=' ',header='L='+str(L)+' N_run='+str(N_run)+'; T M E C Chi')

    print('run data saved as {} \n'.format(fname))
    print(20*'=',' end of train dataset preparation ','='*20)
    print('\n')




elif operation=="G":
    # making test dir
    test_dir = os.path.join(data_dir,'test')

    if os.path.isdir(test_dir):
        shutil.rmtree(test_dir)
        os.mkdir(test_dir)
    else:
        os.mkdir(test_dir)

    ### begin calculation ##########
    print(20*'=',' Sampling in progress ','='*20)
    startime = localtime()
    os.chdir(test_dir)

    p = Pool(processes=cpu_count()-4)
    ctn = p.map(kernel, T_range)
    p.close()

    ctn = np.array(ctn)

    endtime = localtime()
    duration = str(timedelta(seconds=mktime(endtime) - mktime(startime)))
    print('code started on :',strftime('%x %X',startime),'\ncode ended on :',strftime('%x %X',endtime),'\ntime elapsed :',duration)
    print(20*'=',' end of Sampling ','='*20)
    print('\n')


    ############# generate test data ###################
    print(20*'=',' Building test datasets ','='*20)
    
    os.chdir(data_dir)
    file_list,labels_temp,labels = make_labels(test_dir)

    fname = os.path.join(data_dir,'test_dataset.npz')
    np.savez(fname , x_test=file_list,x_temp=labels_temp)
    print('dataset saved as: ', fname)
    print('with labels: ',np.load(fname).files)
    print('note: x is a list of referenced data filenames. \nFor training, please build pipeline that take referenced files into the NN.\n')


    fname = os.path.join(data_dir,'run_gen_data.txt')
    np.savetxt(fname,ctn,delimiter=' ',header='L='+str(L)+' N_run='+str(N_run)+'; T M E C Chi')

    print('run data saved as {} \n'.format(fname))
    print(20*'=',' end of test dataset preparation ','='*20)
    print('\n')


else:
    print('invalid operation, Program quited !')




