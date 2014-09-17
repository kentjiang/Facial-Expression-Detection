%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Facedetection function
% Xinyun Jiang
% ECE 681
% Project Name: Fatical Expression detection
% Mar 11 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Face = facedetection(I)

%%%%%%%%%%%%%%% Image Resize%%%%%%%%%%%%%%%%%%%%%
I=imresize(I,[375,300]);
Faces=[];
numFaceFound=0;
I=double(I);
H=size(I,1);
W=size(I,2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%% EXTRACT SKIN %%%%%%%%%%%%%%%%%%%%%
YCbCr=rgb2ycbcr(I);
Cr=YCbCr(:,:,3);
S=zeros(H,W);
[SkinX,SkinY] =find(10<Cr & Cr<255);
for i=1:length(SkinX)
    S(SkinX(i),SkinY(i))=1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%% REMOVE NOISE %%%%%%%%%%%%%%%%%%%
S=renoise1(S);
S=imfill(S,'holes'); % Fill the holes in the picture, mainly the eyes

%%%%%%%%%%%%%%% Get Face Part %%%%%%%%%%%%%%%%%%%%

[L,lenRegions] = bwlabel(S,4);
AllDat  = regionprops(L,'BoundingBox','FilledArea');
AreaDat = cat(1, AllDat.FilledArea);
[maxArea, maxAreaInd] = max(AreaDat);

FaceDat = AllDat(maxAreaInd);
FaceBB = [FaceDat.BoundingBox(1),FaceDat.BoundingBox(2),...
    FaceDat.BoundingBox(3)-1,FaceDat.BoundingBox(4)-1];

Face=imcrop(rgb2gray(uint8(I)).*uint8(S),FaceBB);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%