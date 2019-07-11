


Ktr=0;Kte=0;  % Ktr denotes training features extracted from hundreds of subnetwork nodes. 
%%%%%%%%%%%%%%%  Kte denotes testing features extracted from hundreds of subnetwork nodes.

for loop=1:12   %% loop denotes data-channel
    %%%%%%%%%%%%%%%%%% SIFT Features loading %%%%%%%%%%%%%%%%%%%%%%%%%
 if loop<=3
     load Caltech_101.mat    %%%%  SIFT features for C101 loading, please download it from google drive£º https://drive.google.com/open?id=0B0amrBuf2cGUZk9kNVd3MGM3TWM

database1=[tr_label tr_fea];
clear tr_fea
clear tr_label
database2=[ts_label ts_fea];
clear ts_fea
clear ts_label
fae=[database1; database2];
clear database1
clear database2
train_per_image=30;  %Training image per class
    sample_G;
    clear fae
clear fea
    Training=tr_fea;   %%%Training features (SIFT) + desired labels
    Testing=ts_fea;    %%%Testing features (SIFT) + desired labels
 end
 
%%%%%%%%%%%%%%%%%%%%% HMP Features loading %%%%%%%%%%%%%%%%%%%%%%%
 
    if loop>3 && loop<=7 
load caltech101_cvpr13_1000   %%% please download it from google drive: https://drive.google.com/open?id=0B0amrBuf2cGUYW5mWXFKUnpxMTA
Training=[Training(:,1) Training(:,3:end)];   %%% Training features + desired labels
Testing=[Testing(:,1) Testing(:,3:end)];    %%% Testing features + desired labels
    end
%%%%%%%%%%%%%%%%%%%%%%% Spatial pyramid features loading %%%%%%%%%%%%%%%%%%
        if loop>7
load spatialpyramidfeatures4caltech101.mat    %%% please download the Mat file from http://www.umiacs.umd.edu/~zhuolin/LCKSVD/features/spatialpyramidfeatures4caltech101.zip
[labelMat,b]=find(labelMat==1);
fae=[labelMat'; featureMat]';
clear featureMat
train_per_image=30;   %Training image per class
    sample_G;
    clear fae
    Training=tr_fea;   %%% Training features + desired labels
    Testing=ts_fea;    %%% Testing features + desired labels
  clear fea
        end
%%%%%%%%%%%%%%%%%%%%% Layer 1: subspace feature extraction %%%%%%%%%%%%%%%%

name=sprintf('C101_channel_%d',loop);
num_subnetwork_node=3;  %%%%% 3 subnetework nodes used %%%%%%%%%%%%
dimension=100;   %%%%%%%%%%%% dimensionality (100) used in each subnetwork node %%%%%%%%%%%%%
C1=2^0;      %%%%%% parameter C in equation (5) %%%%%%%%%%%

[train_time,NumberofTrainingData]=Layerfirst(Training,Testing,1,dimension,'sine',C1,2,num_subnetwork_node,name);   %%%%% subspace features $i$ will be saved as 'C101_subspae_features_i.mat'

%%%%%%%%%%%%%%%%%%%%%%%% Layer 2: kernel combination %%%%%%%%%%%%%%%%%%%%
[Ktr, Kte] = featurecomb(Ktr, Kte,name,num_subnetwork_node,NumberofTrainingData);


end
%%%%%%%%%%%%%%%%%%%%%%%%%% Layer 3: classifier with subnetwork nodes %%%%%%

Training=[Training(:,1) Ktr];   %%%%%% here Ktr denotes final features for training 
 Testing=[Testing(:,1) Kte];   %%%%%% Kte denotes final feature for testing 
 
 C2=2^-13;    %%% Parameter C in equation (15)
[train_time,  train_accuracy11,test_accuracy]=lastlayer(Training,Testing,1,1,'sig',1,C2);

test_accuracy




 







