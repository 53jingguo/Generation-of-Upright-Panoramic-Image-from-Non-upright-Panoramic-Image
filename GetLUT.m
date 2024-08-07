close all;clear;clc
S='bs_1.jpg';% read any one image
f=imread(S); % S is a string (image's name)1024*512
f=imresize(f,[256,512]);   % if the origional image is too large, you can resize the image. ****Maybe you don't need it****
[h,w,td]=size(f);
rImg=zeros(h,w,td);
pai=3.14159;
kk=1;

for rollAngP=0     %Pitch看论文范围如何确定
    for rollAngR=0%:10:180  %Roll
        for rollAngY=0%:10:180 %Yaw 
            LUT=zeros(h*w,2);
            kk=1;
            for i=1:h
                for j=1:w
                    theta=i*180/h*(pai/180);
                    fai=j*360/w*(pai/180);
                    X=sin(theta)*cos(fai);
                    Y=sin(theta)*sin(fai);
                    Z=cos(theta);
                    yaw=rollAngY*(pai/180);
                    pitch=rollAngP*(pai/180);
                    roll=rollAngR*(pai/180);
                    Rx=[1 0 0;0 cos(roll) -sin(roll);0 sin(roll) cos(roll)];
                    Ry=[cos(pitch) 0 sin(pitch);0 1 0;-sin(pitch) 0 cos(pitch)];
                    Rz=[cos(yaw) -sin(yaw) 0;sin(yaw) cos(yaw) 0;0 0 1];
                    R=Rx*Ry*Rz;%以前顺序乘反了
                    Cor_new=R*[X Y Z]';
                    new_X=Cor_new(1);
                    new_Y=Cor_new(2);
                    new_Z=Cor_new(3);
          
                    if new_Z>0
                        theta_new=atan(sqrt(new_X.^2+new_Y.^2)/new_Z);
                    else
                        theta_new=pai-atan(-sqrt(new_X.^2+new_Y.^2)/new_Z);
                    end
                    if new_X>0&&new_Y>0
                        fai_new=atan(new_Y/new_X);
                    end
                    if new_X<0&&new_Y>0
                        fai_new=pai+atan(new_Y/new_X);
                    end
                    if new_X<0&&new_Y<0
                        fai_new=pai+atan(new_Y/new_X);
                    end
                    if new_X>0&&new_Y<0
                        fai_new=2*pai-atan(-new_Y/new_X);
                    end
                    x_new=floor(fai_new*w/360*180./pai);
                    if x_new<=0
                        x_new=1;
                    end
                    y_new=floor(theta_new*h/180*180./pai);
                    if y_new<=0
                        y_new=1;
                    end
                    rImg(i,j,:)=f(y_new,x_new,:);
                    x_new = (x_new - 256.0)/256.0;
                    y_new = (y_new - 128.0)/128.0;
                    LUT(kk,:)=[y_new,x_new];
                    kk=kk+1;
                end
            end
            rImg=uint8(rImg);
            subplot(211),imshow(f)
            subplot(212),imshow(rImg)
            imwrite(rImg,'P70Y90rimg.bmp')
            RollAng=num2str(rollAngR);
            PitchAng=num2str(rollAngP);
            YawAng=num2str(rollAngY);
            Angle=[rollAngP rollAngR];
            NameR60D10(kk,:)=Angle;
            str=strcat('C:\Users\ASUS\Desktop\\LUT','_P',PitchAng,'_R',RollAng,'_Y',YawAng,'.mat');
            save(str,'LUT')
kk=kk+1;
        end
    end
end