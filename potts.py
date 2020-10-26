import numpy as np
import matplotlib.pyplot as plt


class Potts_model2D ():
    def __init__ (self,L,q,alinged=False):
        self.L = L
        self.q = q
        self.q_set = np.arange(self.q+1,dtype=np.int32)[1:]
        self.alinged = alinged
        self.shp = (self.L,self.L)
        self.spin_config_init()


    def spin_config_init(self):
        if self.alinged==False:
            self.spin_config = np.random.choice(self.q_set,size=self.shp)

        if self.alinged==True:
            self.spin_config = np.ones(shape=self.shp,dtype=np.int32)


    def magnetization(self):
        '''
        seek for spins which equal to 1
        '''
        magnetization = 0
        for i in self.spin_config.reshape(self.L*self.L): # create a view of lattice only (shallow-copy)
            if i==1:
                magnetization +=1
        return magnetization
        #return np.sum(self.spin_config) # ???????????????


    def pbc (self,idx):
        if idx == self.L-1:
            return 0
        else:
            return idx+1


    def energy_site(self,spin,i,j):
        sum_of_neighbours = 0
        for neighbour in [self.spin_config[i-1,j], self.spin_config[self.pbc(i),j], self.spin_config[i,j-1], self.spin_config[i,self.pbc(j)]]:
            if spin == neighbour:
                sum_of_neighbours +=1

        #print('sum potts (how many same) = ',sum_of_neighbours) # for debug
        return -sum_of_neighbours



    def energy_total(self):
        result = 0
        for i in range(self.L):
            for j in range(self.L):
                spin = self.spin_config[i,j]
                result += self.energy_site(spin,i,j) # -ve sign due to -J (ferromagnetic) in the hamitonian

        return result/2 # correct for overcounting due to overlapping


    def update (self,T):
        """determine if a spin is updated according to metropolis monte-carlo rules, under a specified temperature T. """
        for i in range(self.L):
            for j in range(self.L):
                spin = self.spin_config[i,j]
                proposed_spin = self.q_set[self.q_set!=self.spin_config[i,j]][np.random.randint(self.q-1)]
                #proposed_spin = np.random.choice(self.q_set[self.q_set!=self.spin_config[i,j]])
                denergy = self.energy_site(proposed_spin,i,j)-self.energy_site(spin,i,j)
                if denergy<0:
                    self.spin_config[i,j] = proposed_spin
                else:
                    if np.e**(-denergy/T)>np.random.uniform(0,1):
                        self.spin_config[i,j] = proposed_spin
                #print('spin is:',self.spin_config[i,j],' set is: ',self.q_set[self.q_set!=self.spin_config[i,j]],' selected: ',selected_state) #for debug


    def batch_flip (self,N_run,N_ss,T):
        """
        the main calculation routine to calculate the values
        
        parameters
        N_run: total number of complete monte-carlo steps (sweep through the lattice)
        N_ss: total number of complete MCS before sampling starts
        T: temperature
        """
        M=0
        m=0
        M_sq=0
        E=0
        e=0
        E_sq=0

        ### stepping only lattice; no data sampling #####
        for k in range (N_ss):
            self.update(T)

        count = 0
        for k in range (N_ss,N_run):
            self.update(T)
            m= self.magnetization()
            M+=m
            M_sq+=m**2
            e=self.energy_total()
            E+=e
            E_sq+=e**2

            fname = '{:.2f}i{}.npy'.format(round(T,2),count)
            np.save(fname,self.spin_config)
            count += 1
        
        volume = self.L**2
        M_mean=M/(N_run-N_ss)
        M_var=M_sq/(N_run-N_ss)-M_mean**2
        E_mean=E/(N_run-N_ss)
        E_var=E_sq/(N_run-N_ss)-E_mean**2
        C=E_var/T**2/volume
        Chi=M_var/T/volume

        return T, M_mean/volume , E_mean/volume ,C , Chi # temporary take out lattice, which is global 


























