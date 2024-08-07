close all;clear;clc
for ii=1:1
    k=0;
    S0='C:\Users\ASUS\Desktop\ImageRotate321\bs_'; % Image Name
    strNub=num2str(ii);
    Form='.jpg';  % format
    S=strcat(S0,strNub,Form); % 'testimg.jpg'
    S1='C:\Users\ASUS\Desktop\ImageRotate321\';
    f=imread(S); % S is a string (image's name)
    f=imresize(f,[256,512]);   % if the origional image is too large, you can resize the image. ****Maybe you don't need it****
    [h,w,td]=size(f);
    rImg=zeros(h,w,td);
    pai=3.14159;
    for rollAngP=0
        for rollAngR=0
            for rollAngY=0
                kk=1;
                LUT=load(strcat('C:\Users\ASUS\Desktop\LUT.mat'));
                for i=1:h
                    for j=1:w
                        rImg(i,j,:)=f(LUT(kk,0),LUT(kk,1),:);
                        kk=kk+1;
                    end
                end
                rImg=uint8(rImg);
                RollAng=num2str(rollAngR);
                PitchAng=num2str(rollAngP);
                YawAng=num2str(rollAngY);
                saveNub=num2str(k);
                saveRoot=strcat(S1,saveNub,'/');
                mkdir(saveRoot)
                str=strcat(saveRoot,strNub,'.jpg');
                imwrite(rImg,str)
                k=k+1;
            end
        end
    end
end