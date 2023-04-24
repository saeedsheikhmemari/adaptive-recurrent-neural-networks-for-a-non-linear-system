%RNN Controller For Nonlinear System

clc;
clear;
close all;
alpha=0.01;
itNN=180;
n=itNN-1;
Neuron=10;    %Number of neurons in hidden layers
NumInt=8;    %Number of inputs

Theta1=zeros(1,itNN);
% input
ref1=randi([60 60] ,1, 60);
ref2=randi([20 20] ,1, 60);
ref22=randi([40 40],1, 60);
ref3=[ref1 ref2 ref22];

%reference Model
refe1=randi([60 60] ,1, 60);
refe2=randi([20 20] ,1, 60);
refe22=randi([40 40],1, 60);
reference=[refe1 refe2 refe22];

%%secound input
% tt=linspace(-10,10,180);
% ref3=sin(tt);


%reference Model 2
% tt=linspace(-10,10,180);
% reference=sin(tt);

%% Initialize DRNN Controller

        
% Weights
W_w.w1=[];        % Weights From inputs to  Hidden layer1
W_w.w2=[];        % Weights From Hidden layer1 to Hidden layer2
W_w.w3=[];        % Weights From Hidden layer2 to Hidden layer3
W_w.w4=[];        % Weights From Hidden layer3 to output node


Ww=repmat(W_w,itNN ,1);

for k=1:itNN
    
    Ww(k).w1=unifrnd(0,1,NumInt,Neuron);
    Ww(k).w2=unifrnd(0,1,Neuron,Neuron);
    Ww(k).w3=unifrnd(0,1,Neuron,Neuron);
    Ww(k).w4=unifrnd(0,1,1,Neuron);
    
end

E=zeros(1,itNN);                              %Error between Theta and reference
E1=zeros(1,itNN);                             % observing and reduce Error
E2=zeros(1,itNN);
u=zeros(1,itNN);                                %NN OutPut

for it=1:3000
for i=3:n

  Input_of_Hidden_layer1 =Ww(i).w1'*[ref3(i) E1(i) E1(i-1) E1(i-2) Theta1(i) Theta1(i-1) u(i-1) u(i)]'+1;


    output_ofHidden_layer1 =  1./(1+exp(-Input_of_Hidden_layer1));
     
     Input_of_Hidden_layer2 = (Ww(i).w2*output_ofHidden_layer1)+1;
     
     output_ofHidden_layer2 =  1./(1+exp(-Input_of_Hidden_layer2));
     
     Input_of_Hidden_layer3 = (Ww(i).w3*output_ofHidden_layer2)+10;
     
     output_ofHidden_layer3 =  1./(1+exp(-Input_of_Hidden_layer3));
     
     Input_of_output_Node =   (Ww(i).w4*output_ofHidden_layer3)+1;
     
     u(i) =Input_of_output_Node; % out put of output layer(linear)
     if u(i)>10
         u(i)=10;
     elseif u(i)<0
         u(i)=0;
     end
         

     Theta1(i+1)=0.9*Theta1(i)-0.001*Theta1(i-1)^2+u(i)+sin(u(i-1));
     
% Back Propagation

     E1(i)=(ref3(i+1)-Theta1(i+1));
     E2(i)=(reference(i+1)-Theta1(i+1));
     E(i)=(E2(i)-u(i));
     

     error_of_hidden_layer3=Ww(i).w4'*E2(i);
   
     Delta3=(Input_of_Hidden_layer3>0).*error_of_hidden_layer3;
    
     error_of_hidden_layer2=Ww(i).w3'*Delta3;
     Delta2=(Input_of_Hidden_layer2>0).*error_of_hidden_layer2;
    
     error_of_hidden_layer1=Ww(i).w2'*Delta2;
     Delta1=(Input_of_Hidden_layer1>0).*error_of_hidden_layer1;
    
     adjustment_of_W4= alpha*E2(i) * output_ofHidden_layer3';
     adjustment_of_W3= alpha*Delta3 * output_ofHidden_layer2';
     adjustment_of_W2= alpha*Delta2* output_ofHidden_layer1';
     adjustment_of_W1= alpha*Delta1*[ref3(i) E1(i) E1(i-1) E1(i-2) Theta1(i) Theta1(i-1) u(i-1) u(i)];
    
      %Transpose adjusted Weights if inputs elements are not equals to hidden layer nodes
    adjustment_of_W1=adjustment_of_W1';
    %uptodata Weights
    Ww(i).w1=Ww(i).w1+adjustment_of_W1;
    Ww(i).w2=Ww(i).w2+adjustment_of_W2;
    Ww(i).w3=Ww(i).w3+adjustment_of_W3;
    Ww(i).w4=Ww(i).w4+adjustment_of_W4;


end
end

%% Results

 % Result 
 figure;
  plot(reference,'LineWidth',2)   
  hold on;
  plot(Theta1,'r','LineWidth',1.2)
  plot(u)
  legend('reference','system output','input signal')
  title(' NN Controler')
  hold off
  
