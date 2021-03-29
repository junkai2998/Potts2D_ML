############# q-states Potts model via monte-carlo simulation #############
there are slight modification, which we only train on extreme temperatures

(A) Components:
1. 00sampling.ipynb: contain the core codes to run simulation and generate data. Includes data visualization for run data as well.


2. potts.py: library for Potts_model2D class. Outputs:
            a) data[date][q-value]/train folder
            b) run_data.txt that contains averaged thermodynamic quantities at temperatures
            c) train_data.npz that contains the file names in train folder with labels
               * core code from 3DIsing_v1.1/mc_class.py. Metropolis MC code modified to suite q-state operations.

3. run.py: an AIO CLI code for train/test data generation. Outputs:
            a) data[date][q-value]/test folder
            b) run_gen_data.txt that contains averaged thermodynamic quantities at temperatures
            c) test_data.npz that contains the file names in test folder with labels
               * core code from core code from 3DIsing_v1.1/RUN.py. Modified to print/store q-value.

4. 01visualize.ipynb: run through samples to visualize spin configurations.

5. 02train.ipynb

6. 03predict.ipynb



(C) Known Issues
1. Forgot to normalize the values in the spin configuration to (0,1) for train/test during training / validating. But still gives reasonable result.
   solved. But prediction results behave slightly different (curve slope) near Tc.


(D) Future work
1. Try Swendsen-Wang / Wolff algorithm

nohup python potts/run.py -L 40 -q 4 -N_run 2000 -fracN_ss 0.5 -Tini 0.0 -Tlast 2.0 -dt 0.05 G