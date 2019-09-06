#!/usr/bin/octave-cli --persist
close all
pkg load optim;

############ For UI
clear h
graphics_toolkit qt


############ Gradients


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
 
function [ts, es]=gradientDesBatch(t, X, Y, alpha)
  # Perform the gradient descent (batch)
  ts=t; # sequence of t's
  es=[]; # for error
  for i=[1:200] # max 200 iterations
    tc = ts(rows(ts),:); # Current position 
    gn = gradJ(tc,X,Y);  # Gradient at current position
    e = J(tc, X, Y); 
    tn = tc - alpha * gn;# Next position
    ts = [ts;tn];
    
    es=[es;e];
    if (norm(tc-tn)<0.001) break; endif;
  endfor
endfunction

function [ts, es]=gradientDesStoch(t, X, Y, alpha)
  # Perform the gradient descent (stochastic)
  ts=t; # sequence of t's
  es=[]; # sequence of errors
  j=0;
  for i=[1:4000] # max 4000 iterations
    tc = ts(rows(ts),:); # Current position
    sample=round(unifrnd(1,rows(X))); # Use one random sample
    gn = gradJ(tc,X(sample,:),Y(sample));  # Gradient at current position
    tn = tc - alpha * gn;# Next position
    ts = [ts;tn];
    e = J(tc, X, Y);
    es =[es;e];
    if (norm(tc-tn)<0.0001)
      j=j+1;
      if (j>5)
        break;
      endif;
    else
      j=0;
    endif;
  endfor
endfunction




############ Main program

function gradientDescent(obj)
  
  h = guidata(obj);
  #For only one decimal place
  x = round(10 * get(h.x_slider, "value")) /10;
  y = round(10 * get(h.y_slider, "value")) /10;
  z = round(10 * get(h.z_slider, "value")) /10;
  
  #Initial point
  t = [x y z];
  alpha = 0.01;
  
  ############ Data

  # Data stored each sample in a row, where the last row is the label
  D=load("escazu32.dat");
  # Construct the design matrix with the original data
  Xo=[ones(rows(D),1),D(:,1)];
  # The outputs vector with the original data
  Yo=D(:,4);


  ############ Normalization


  # The slope for the data normalization
  minArea = min(Xo(:,2));
  maxArea = max(Xo(:,2));
  mx = 2/(maxArea-minArea);
  bx = 1-mx*maxArea;

  NM=[1 bx; 0 mx];
  X=Xo*NM; # Normalized data to interval -1 to 1

  # Add the third column with x_1^2 to apply as if it was linear regression
  X(:,3) = X(:,2).^2;

  # Normalize also the output
  minPrice=min(Yo);
  maxPrice=max(Yo);
  my = 2/(maxPrice-minPrice);
  by = 1-my*maxPrice;

  Y = my*Yo + by;

  # For the inverse mapping we need these:
  imy = 1/my;
  iby = -by/my;


  ############ Plotting

  [ts, es] = gradientDesBatch(t, X ,Y, alpha);
  [tss, ess] = gradientDesStoch(t, X ,Y, alpha);

  ############ Batch trajectory

  figure(2, "name", "Batch Trajectory");
  hold off;
  plot3(ts(:,1),ts(:,2),ts(:,3),"k-");
  hold on;
  plot3(t(1),t(2),t(3),"*r");
  plot3(ts(:,1),ts(:,2),ts(:,3),"ob");
  box on;

  title("Batch");
  xlabel('\theta_0');
  ylabel('\theta_1');
  zlabel('\theta_2');
  xlim([-1 0]);
  ylim([-0.2 1.4]);
  zlim([-0.4 0.4]);

  ############# Stochastic trajectory
  figure(3, "name", "Stochastic Trajectory");
  hold off;
  plot3(tss(:,1),tss(:,2),tss(:,3),"k-");
  hold on;
  plot3(tss(:,1),tss(:,2),tss(:,3),"ob");
  plot3(t(1),t(2),t(3),"*r");
  box on;



  title("Stochastic");
  xlabel('\theta_0');
  ylabel('\theta_1');
  zlabel('\theta_2');
  xlim([-1 0]);
  ylim([-0.2 1.4]);
  zlim([-0.4 0.4]);

  ############## Batch curves

  figure(4, "name", "Batch Curves")
  hold off;
  plot(D(:,1), D(:,4),"marker", "*", "color", "b", "LineStyle", "none");
  hold on;

  # The line back in the samples
  areas=linspace(minArea,maxArea,10);
  # Normalize these input areas
  areasNorm=mx*areas+bx;
  # Calculate values with those normalized areas
  pricesNorm = ts*[ones(size(areasNorm));areasNorm;areasNorm.^2];
  # We have to de-normalize the normalized estimation
  prices=imy*(pricesNorm)+iby;
  # First plot first black
  plot(areas,prices(1,:),'k',"linewidth",3);
  # Now plot intermediate
  for i=linspace(2, rows(prices)-1, 100)
    plot(areas,prices(round(i),:),'c',"linewidth",1);
  endfor
  # Finally plot last one
  plot(areas,prices(end,:),'r',"linewidth",3);

  title("Batch curves");
  xlabel("area");
  ylabel("price");
  xlim([minArea 600]);
  ylim([0 800]);


  ############# Stochastic curves

  figure(5, "name", "Stochastic Curves");
  hold off;
  plot(D(:,1), D(:,4),"marker", "*", "color", "b", "LineStyle", "none");
  hold on;

  # The line back in the samples
  areas=linspace(minArea,maxArea,10);

  # Normalize these input areas
  areasNorm=mx*areas+bx;

  # Calculate values with those normalized areas
  pricesNorm = tss*[ones(size(areasNorm));areasNorm;areasNorm.^2];

  # We have to de-normalize the normalized estimation
  prices=imy*(pricesNorm)+iby;

  # First plot first black
  plot(areas,prices(1,:),'k',"linewidth",3);

  # Now plot intermediate
  for i=linspace(2, rows(prices)-1, 100)
    plot(areas,prices(round(i),:),'c',"linewidth",1);
  endfor
  # Finally plot last one
  plot(areas,prices(end,:),'r',"linewidth",3);

  title("Stochastic curves");
  xlabel("area");
  ylabel("price");
  xlim([minArea 600]);
  ylim([0 800]);

  ###################Error in Batch descent method

  figure(6, "name", "Batch Error")

  hold off;
  #alpha = 0.001
  [ts, es] = gradientDesBatch(t, X ,Y, 0.001);
  plot(1:size(es), es, 'k',"linewidth",3);
  hold on;
  #alpha = 0.005
  [ts, es] = gradientDesBatch(t, X ,Y, 0.005);
  plot(1:size(es), es, 'r',"linewidth",3);
  #alpha = 0.01
  [ts, es] = gradientDesBatch(t, X ,Y, 0.01);
  plot(1:size(es), es, 'g',"linewidth",3)
  #alpha = 0.045
  [ts, es] = gradientDesBatch(t, X ,Y, 0.045);
  plot(1:size(es), es, 'b',"linewidth",3);
  #alpha = 0.0494
  [ts, es] = gradientDesBatch(t, X ,Y, 0.0494);
  plot(1:size(es), es, 'm',"linewidth",3);
  legend({'\alpha=0.001', '\alpha=0.005', '\alpha=0.01','\alpha=0.045', '\alpha=0.0494'});

  title("Batch error");
  xlabel("Iteration");
  ylabel("Error");
  xlim([0 200]);
  ylim([0 12]);
  hold off



  ###################Error in Stochastic descent method

  figure(7, "name", "Stochastic Error")

  hold off;
  #alpha = 0.001
  [ts, es] = gradientDesStoch(t, X ,Y, 0.001);
  plot(1:size(es), es, 'k',"linewidth",3);
  hold on;
  #alpha = 0.005
  [ts, es] = gradientDesStoch(t, X ,Y, 0.005);
  plot(1:size(es), es, 'r',"linewidth",3);
  #alpha = 0.01
  [ts, es] = gradientDesStoch(t, X ,Y, 0.01);
  plot(1:size(es), es, 'g',"linewidth",3)
  #alpha = 0.045
  [ts, es] = gradientDesStoch(t, X ,Y, 0.045);
  plot(1:size(es), es, 'b',"linewidth",3);
  #alpha = 0.0494
  [ts, es] = gradientDesStoch(t, X ,Y, 0.0494);
  plot(1:size(es), es, 'm',"linewidth",3);
  legend({'\alpha=0.001', '\alpha=0.005', '\alpha=0.01','\alpha=0.045', '\alpha=0.0494'});

  title("Stochastic error");
  xlabel("Iteration");
  ylabel("Error");
  hold off

endfunction


################# GUI

function updateLabels(obj)
  h = guidata(obj);
  switch (gcbo)
    case {h.x_slider}
      v = get(h.x_slider, "value");
      set (h.x_label, "string", sprintf("X: %.1f", v));
    case {h.y_slider}
      v = get(h.y_slider, "value");
      set (h.y_label, "string", sprintf("Y: %.1f", v));
    case {h.z_slider}
      v = get(h.z_slider, "value");
      set (h.z_label, "string", sprintf("Z: %.1f", v));
   endswitch
endfunction


h.title_label = uicontrol ("style", "text",
                           "units", "normalized",
                           "string", "Select the starting point:",
                           "horizontalalignment", "left",
                           "position", [0 0.7 0.3 0.5]);

## calculate gradient
h.plot_pushbutton = uicontrol ("style", "pushbutton",
                                "units", "normalized",
                                "string", "Calculate Gradient",
                                "callback", @gradientDescent,
                                "position", [0.1 0.1 0.8 0.09]);
                             
                                
h.x_label = uicontrol ("style", "text",
                           "units", "normalized",
                           "string", "X: -1",
                           "horizontalalignment", "left",
                           "position", [0.45 0.7 0.8 0.08]);

h.x_slider = uicontrol ("style", "slider",
                            "units", "normalized",
                            "string", "X",
                            "value", -1,
                            "min", -1,
                            "max", 0,
                            "sliderstep", [0.1 0.1],                      
                            "callback", @updateLabels,
                            "position", [0.3 0.65 0.4 0.06]);

h.y_label = uicontrol ("style", "text",
                           "units", "normalized",
                           "string", "Y: -0.2",
                           "horizontalalignment", "left",
                           "position", [0.45 0.5 0.8 0.08]);

h.y_slider = uicontrol ("style", "slider",
                            "units", "normalized",
                            "string", "Y",
                            "value", -0.2,
                            "min", -0.2,
                            "max", 1.4,
                            "callback", @updateLabels,
                            "position", [0.3 0.45 0.4 0.06]);

h.z_label = uicontrol ("style", "text",
                           "units", "normalized",
                           "string", "Z: -0.3",
                           "horizontalalignment", "left",
                           "position", [0.45 0.3 0.8 0.08]);

h.z_slider = uicontrol ("style", "slider",
                            "units", "normalized",
                            "string", "Z",
                            "value", -0.3,
                            "min", -0.4,
                            "max", 0.4,
                            "callback", @updateLabels,
                            "position", [0.3 0.25 0.4 0.06]);


set (gcf, "color", get(0, "defaultuicontrolbackgroundcolor"))
figure(1, "name", "Interactive Gradient Descent");
guidata (gcf, h)
