
pkg load optim;
# Data stored each sample in a row, where the last row is the label
D=load("escazu32.dat");


#Construct the matrix powering x from 0 to 2 in each column with the original data.
Xo = bsxfun(@power, D(:,1), 0:1);

# The outputs vector with the original data
Yo=D(:,4);

# The slope for the data normalization
minArea = min(Xo(:,2));
maxArea = max(Xo(:,2));
mx = 2/(maxArea-minArea);
bx = 1-mx*maxArea;

NM=[1 bx; 0 mx];
X=Xo*NM; # Normalized data to interval -1 to 1

X(:,3) = X(:,2).^2;

# Normalize also the output
minPrice=min(Yo);
maxPrice=max(Yo);
my = 2/(maxPrice-minPrice);
by = 1-my*maxPrice;

Y = my*Yo + by;

# Gradient of J.
# Analytical solution.
#
# Here assuming that theta has two components only
# For each theta pair (assumed in a row of the theta matrix) it will
# compute also a row with the gradient: the first column is the partial
# derivative w.r.t theta_0 and the second w.r.t theta_1
function res=gradJ(theta,X,Y)
  res=(X'*(X*theta'-Y*ones(1,rows(theta))))';
endfunction;

# Learning rate
alpha = 0.005;

t=[-1 -0.2 -0.3];
gt=gradJ(t,X,Y);

t0 = -1;
t1 = -0.2;
t2 = -0.3;
 

# Perform the gradient descent (batch)
ts=t; # sequence of t's

for i=[1:100] # max 100 iterations
  tc = ts(rows(ts),:); # Current position 
  gn = gradJ(tc,X,Y);  # Gradient at current position
  tn = tc - alpha * gn;# Next position
  ts = [ts;tn];


  if (norm(tc-tn)<0.001) break; endif;
endfor

# Perform the gradient descent (stochastic)
tss=t; # sequence of t's

j=0;
for i=[1:4000] # max 100 iterations
  tcs = tss(rows(tss),:); # Current position
  sample=round(unifrnd(1,rows(X))); # Use one random sample
  gns = gradJ(tcs,X(sample,:),Y(sample));  # Gradient at current position
  tns = tcs - alpha * gns;# Next position
  tss = [tss;tns];

  if (norm(tcs-tns)<0.0001)
    j=j+1;
    if (j>5)
      break;
    endif;
  else
    j=0;
  endif;
endfor


figure(1);
plot3(ts(:,1),ts(:,2),ts(:,3),"k-");
hold on;
plot3([t0],[t1],[t2],"*r");
plot3(ts(:,1),ts(:,2),ts(:,3),"ob");
box on;

title("Batch");
xlabel("theta_0");
ylabel("theta_1");
zlabel("theta_2");
xlim([-1 0]);
ylim([-0.2 1.4]);
zlim([-0.4 0.4]);

figure(2);
plot3(tss(:,1),tss(:,2),tss(:,3),"k-");
hold on;
plot3(tss(:,1),tss(:,2),tss(:,3),"ob");
plot3([t0],[t1],[t2],"*r");


box on;

title("Stochastic");
xlabel("theta_0");
ylabel("theta_1");
zlabel("theta_2");
xlim([-1 0]);
ylim([-0.2 1.4]);
zlim([-0.4 0.4]);

