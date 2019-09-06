#############################################
###            DATA LOADING               ###
#############################################

load quasar_train.csv;
lambdas = quasar_train(1, :)';
train_qso = quasar_train(2:end, :);
load quasar_test.csv;
test_qso = quasar_test(2:end, :);


#############################################
###            LINEAR REGRESSION          ###
#############################################

Xo = [ones(rows(lambdas),1),lambdas];
Yo = train_qso(1,:)';

# The slope for the data normalization
minArea = min(Xo(:,2));
maxArea = max(Xo(:,2));
mx = 2/(maxArea-minArea);
bx = 1-mx*maxArea;

NM=[1 bx; 0 mx];
X=Xo*NM; # Normalized data to interval -1 to 1

# Normalize also the output
minPrice=min(Yo);
maxPrice=max(Yo);
my = 2/(maxPrice-minPrice);
by = 1-my*maxPrice;

Y = my*Yo + by;

# For the inverse mapping we need these:
imy = 1/my;
iby = -by/my;

# Normal equation
theta = pinv(X)*Y;

figure(1, "name", "Linear regression");
hold off;
plot(Xo(:,2),Yo,"*b");
hold on;

# We have to de-normalize the normalized estimation
lums=theta(2)*imy*mx*lambdas + (imy*theta(2)*bx + imy*theta(1)+iby);
plot(lambdas,lums,'k',"linewidth",3);
title("Linear regression");
legend({'Raw data','Linear Regression'});
xlabel('\lambda');
ylabel('Flux');

#############################################
###     LOCALLY WEIGHTED REGRESSION       ###
#############################################

figure(2, "name", "Locally weighted regression");

hold off;
plot(Xo(:,2),Yo,"*b");
hold on;

colors = ["r", "b", "g", "m", "c"]; 
tau = [1, 5, 10, 100, 1000];
exponent_tau = 1./(tau.^2);
num_tests = size(lambdas);

# Normalize the lambdas
minlambda=min(lambdas);
maxlambda=max(lambdas);
my = 2/(maxlambda-minlambda);
by = 1-my*maxlambda;

%lambdas normalized
normalized_lambdas = my*lambdas' + by;
%constant part of the exponent
exponent = e.^(-(((lambdas' - lambdas).^2)./(2)));

for (j=[1:columns(tau)])

  %weights
  wi = exponent.^(exponent_tau(j));

  %Initialize results normalized
  lums_normalized = zeros(1,num_tests);

  for i=[1:num_tests]

    %Diagonal matrix with weights
    W = diag(wi(:,i)); 

    %Demonstrated equation
    THETA = inv(X'*W*X) * X'*W*Y;

    lums_normalized(i) = THETA'*[1; normalized_lambdas(i)];

  endfor

  lums=imy*lums_normalized + iby;
  plot(lambdas, lums, colors(j), "linewidth", 3);

endfor
title("Locally weighted regression");
legend({'Raw data','\tau=1','\tau=5', '\tau=10', '\tau=100', '\tau=1000'});
xlabel('\lambda');
ylabel('Flux');





