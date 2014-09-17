%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main
% Xinyun Jiang
% ECE 681
% Project Name: Fatical Expression detection
% Mar 11 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
strTrainPath = 'Train';
strLabel = 'Label.txt';
strTestPath = 'Test';

fid=fopen(strLabel);
imageLabel=textscan(fid,'%s %s','whitespace',',');
fclose(fid);

NeutralImages=[];
for i=1:length(imageLabel{1,1})
    if (strcmp(lower(imageLabel{1,2}{i,1}),'neutral'))
        NeutralImages=[NeutralImages,i];
    end 
end

structTestImages = dir(strTestPath);
NumImage = length(imageLabel{1,1}); 
lenTest = length(structTestImages);

TrainImages='';
for i = 1:NumImage
	TrainImages{i,1} = strcat(strTrainPath,'\',imageLabel{1,1}(i));
end

NumTest=0; % Number of Test Images
for i = 3:lenTest
     if ((~structTestImages(i).isdir))
         if  (structTestImages(i).name(end-3:end)=='.jpg')
             NumTest=NumTest+1;
             TestImages{NumTest,1} = [strTestPath,'\',...
                 structTestImages(i).name];
         end
     end
end
 
clear ('structTestImages','fid','i','j');

ImgSize = [280,180];

%% ################# Load Train Data & Preprocess  ########################
%% Loading training images & preparing for PCA by subtracting mean
NumImg=NumImage;
face = zeros(ImgSize(1)*ImgSize(2),NumImg);
for i = 1:NumImg
    facet = imresize(facedetection(imresize(imread(cell2mat(...
    TrainImages{i,1})),[375,300])),ImgSize);
    face(:,i) = facet(:);
    disp(sprintf('Loading Training Image # %d',i));
end
meanface = mean(face,2);        
meanmeanface=meanface;                 
face = (face - meanface*ones(1,NumImg))';      

%% ########################################################################


%% ################# Low Dimension Face Space Construction ################
[C,S,L]=princomp(face,'econ'); % PCA
EigenRange = [1:20]; % Define Eigenvalues that will be selected
C = C(:,EigenRange);
%% ########################################################################


%% ############# Load Test Data and project on Face Space #################

face = zeros(ImgSize(1)*ImgSize(2),NumTest);
for i = 1:NumTest
    facet = imresize(facedetection(imresize(imread(...
    TestImages{i,1}),[375,300])),ImgSize);
    face(:,i) = facet(:);
    disp(sprintf('Loading Test Image # %d',i));
end
meanImg = mean(face,2);        
face = (face - meanmeanface*ones(1,NumTest))';
ProjectedTest = face*C;

%% ########################################################################


%% ################# Calculation of Distance from Neutral ##################

meanNutral = mean(S(NeutralImages,EigenRange)',2);
for Dat2Project = 1:NumTest
    TestImage = ProjectedTest(Dat2Project,:);
    EuclDist(Dat2Project) = sqrt((TestImage'-meanNutral)'*(TestImage' ...
        -meanNutral));
end

%% ########################################################################

%% ################# Calculation of other Distances #######################

Other_Dist = zeros(NumTest,NumImg);
for Dat2Project = 1:NumTest
    TestImage = ProjectedTest(Dat2Project,:);
    % Picking the image #Dat2Project
    for i = 1:NumImg
        Other_Dist(Dat2Project,i) = sqrt((TestImage'-S(i,EigenRange)')' ...
            *(TestImage'-S(i,EigenRange)'));
    end
end
[Min_Dist,Min_Dist_pos] = min(Other_Dist,[],2);

%% ########################################################################


%% ########################## Display Result ##############################
fid = fopen('Results.txt','w');
fprintf(fid,'Test Image,Distance From Neutral, Expression,Best Match\r\n');

for i = 1:NumTest
    b = find(TestImages{i,1}=='\');
    Test_Image = TestImages{i,1}(b(end)+1:end);
    Dist_frm_Neutral = EuclDist(i);
    Best_Match = cell2mat(imageLabel{1,1}(Min_Dist_pos(i)));
    Expr = cell2mat(imageLabel{1,2}(Min_Dist_pos(i)));
    fprintf(fid,'%s,%0.0f,%s,%s\r\n',Test_Image,Dist_frm_Neutral,Expr,Best_Match);
end
fclose(fid);
%% ########################################################################
disp('Done')
disp('Output File = Results.txt');