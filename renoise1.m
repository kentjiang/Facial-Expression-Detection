%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Noise Remove function
% Xinyun Jiang
% ECE 681
% Project Name: Fatical Expression detection
% Mar 11 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function SN=renoise1(S)
H=375;
W=300;
SN=zeros(375,300);
for i=1:H-5
    for j=1:W-5
        localSum=sum(sum(S(i:i+4, j:j+4)));
        SN(i:i+5, j:j+5)=(localSum>20);
    end
end
end