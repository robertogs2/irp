load quasar_train.csv;
lambdas = quasar_train(1, :)';
train_qso = quasar_train(2:end, :);
load quasar_test.csv;
test_qso = quasar_test(2:end, :);


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

theta = pinv(X)*Y;

figure(1);
hold off;
plot(Xo(:,2),Yo,"*b");
hold on;

# The line back in the samples
lambda_plot=lambdas;%linspace(min(Xo(:,2)),max(Xo(:,2)),5);

# We have to de-normalize the normalized estimation
lums=theta(2)*imy*mx*lambda_plot + (imy*theta(2)*bx + imy*theta(1)+iby);
plot(lambda_plot,lums,'k',"linewidth",3);


tau = [1, 5, 10, 100, 1000];
exponent_tau = 1/(tau(2)^2);

%raw
xio = lambdas;

# Normalize also the output
minlambda=min(xio);
maxlambda=max(xio);
my = 2/(maxlambda-minlambda);
by = 1-my*maxlambda;

xi = my*xio + by;

num_tests = 450;

%lambdas raw
test_lambdaso = lambdas';

%lambdas normalized
test_lambdasi = my*test_lambdaso + by;

% for (i=[1:size(tau)])

% endfor

exponent = e.^(-(((test_lambdaso - xio).^2)./(2)));

wi = exponent.^(exponent_tau);

% W = diag(wi(:,1)); 

% THETA = inv(X'*W*X) * X'*W*Y;

yi = zeros(1,num_tests);

for i=[1:num_tests]

  W = diag(wi(:,i)); 

  THETA = inv(X'*W*X) * X'*W*Y;

  yi(i) = THETA'*[1; test_lambdasi(i)];

endfor
yi;
lumsi=imy*yi + iby;

figure(2)

hold off;
plot(Xo(:,2),Yo,"*b");
hold on;
plot(test_lambdaso, lumsi, "r", "linewidth", 3);