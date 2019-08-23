
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

# Select the grid to plot the contours and gradients
th0=-1:0.05:0;
th1=-0.2:0.05:1.4;
th2=-0.4:0.05:0.4;
% [tt0,tt1,tt2] = meshgrid(th0,th1,th2);
% theta=[tt0(:) tt1(:), tt2(:)];


# Objective function of the parameters theta, requires also the data A
# First create a matrix without the square, where the j-column has
# theta_0 + theta_1*x_1^(j)-y^(j).  Then, square all elements of that matrix
# and finally add up all elements in each row
function res=J(theta,X,Y)
  D=(X*theta'-Y*ones(1,rows(theta)));
  res=0.5*sum(D.*D,1)';
endfunction;

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
 

# Perform the gradient descent
ts=t; # sequence of t's

for i=[1:100] # max 100 iterations
  tc = ts(rows(ts),:); # Current position 
  gn = gradJ(tc,X,Y);  # Gradient at current position
  tn = tc - alpha * gn;# Next position
  ts = [ts;tn]


  if (norm(tc-tn)<0.001) break; endif;
endfor


figure(1);
#plot3(ts(:,1),ts(:,2),ts(:,3),"k-");
plot3(ts(:,1),ts(:,2),ts(:,3),"ob");
   #plot3([t0],[t1],[t2],"*g");
box on

while(1)

 xlabel("theta_0");
 ylabel("theta_1");
 zlabel("theta_2");
 
endwhile;
