

%%%%%%%%%%% Running this code needs 16 GB memory, and 8 to 10 minutes %%%%
clear
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Features pre-loading %%%%%%%%%%%%%%%%%

f1=1:4:40;
f2=2:4:40;
f3=3:4:40;
f4=4:4:40;
load distancematrices102   %%%% please download it from http://www.robots.ox.ac.uk/~vgg/data/flowers/102/distancematrices102.mat
load imagelabels.mat  
for sxx=1:10
data{f1(sxx)}=Dhog;
data{f2(sxx)}=Dhsv;
data{f3(sxx)}=Dsiftbdy;
data{f4(sxx)}=Dsiftint;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Ktr=0;Kte=0;  % Ktr denotes training features extracted from hundreds of subnetwork nodes. 
%%%%%%%%%%%%%%%  Kte denotes testing features extracted from hundreds of subnetwork nodes.

for loop=9:14  %% loop denotes data-channel number
 %%%%%%%%%%%%%%%%%%%%%%%  SIFT+HOG+HSV features loading  %%%%%%%%%%%%%%%%%%
    if loop<=8
    fae=[labels; data{loop}]';
    train_per_image=20;
    sample_G;
    clear fae
Training=tr_fea;
Testing=ts_fea;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
        %%%%%%%%%%%%%% HMP Features loading %%%%%%%%%%
        if loop>8 && loop<=12
load flower102_HMP_28000.mat  %%% please download it from http://www1.uwindsor.ca/engineering/cvss/system/files/Datasets/Yimin_Yang/flower102_HMP_28000.zip
        end
        
        if loop>12
load Flower102_Alex_feature.mat   %%% please download the Mat file from http://www.umiacs.umd.edu/~zhuolin/LCKSVD/features/spatialpyramidfeatures4caltech101.zip
A=unique(testSet.Labels);
for i=1:length(A)
    ind=find(testSet.Labels==A(i));
    test_label(ind)=i;
end

A=unique(trainingSet.Labels);
for i=1:length(A)
    ind=find(trainingSet.Labels==A(i));
    train_label(ind)=i;
end

trainingFeatures=double(trainingFeatures');
testFeatures=double(testFeatures);


    %Training=;   %%% Training features + desired labels
    Testing=[test_label' mapminmax(testFeatures,-1,1)];    %%% Testing features + desired labels
Training=[train_label' mapminmax(trainingFeatures,-1,1)];    %%% Testing features + desired labels
        end
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    
name=sprintf('flower102_channel_%d',loop);
num_subnetwork_node=3;  %%%%% 3 subnetework nodes used %%%%%%%%%%%%
dimension=300;   %%%%%%%%%%%% dimensionality (100) used in each subnetwork node %%%%%%%%%%%%%
C1=2^-2;      %%%%%% parameter C in equation (5) %%%%%%%%%%%

[train_time,NumberofTrainingData]=Layerfirst(Training,Testing,1,dimension,'sine',C1,2,num_subnetwork_node,name); %%%%% subspace features $i$ will be saved as 'flower102_subspace_features_i.mat'


%%%%%%%%%%%%%%%%%%%%%%%% Layer 2: kernel combination %%%%%%%%%%%%%%%%%%%%
[Ktr, Kte] = featurecomb(Ktr, Kte,name,3,NumberofTrainingData);
    end



Training=[Training(:,1) Ktr];   %%%%%% here Ktr denotes final features for training 
 Testing=[Testing(:,1) Kte];   %%%%%% Kte denotes final feature for testing 
  Training(:,2:end)=mapminmax(Training(:,2:end),-1,1);
Testing(:,2:end)=mapminmax(Testing(:,2:end),-1,1);
%%%%%%%%%%%%%%%%%%%%%%%%%% Layer 3: classifier with subnetwork nodes %%%%%%
 C2=2^-13;    %%% Parameter C in equation (15)
[train_time,  train_accuracy11,test_accuarcy]=lastlayer(Training,Testing,1,1,'sig',1,C2);   

test_accuarcy





 







