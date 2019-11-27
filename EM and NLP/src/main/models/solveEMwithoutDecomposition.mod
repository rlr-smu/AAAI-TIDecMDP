param n;
param gamma default 0.8;
set S{1..n} ordered;
set A{1..n} ordered;

set sparseP within {i in 1..n,A[i],S[i],S[i]};
param sparsePVal{sparseP};

param P{i in 1..n, j in A[i], k in S[i], l in S[i]} default 0;
param R{i in 1..n, j in S[i], k in A[i]};
param alpha{i in 1..n, j in S[i]};

set numcons;
set numprims;	
set numevents;
param primitives{i in numprims, j in 1..4};
param creward{i in numcons};
set events{numevents} ordered;
set cons{numcons} ordered;

param Rmax;
param Rmin;
param theta;

param thetahat = (theta) / (Rmax - Rmin);
param Rcap{i in 1..n, j in S[i], k in A[i]} = (R[i,j,k]) / (Rmax - Rmin);
param crewardcap{i in numcons} = (creward[i]) / (Rmax - Rmin);
param iter_value default 0;

param x{i in 1..n, j in S[i], k in A[i]};
var xstar{i in 1..n, j in S[i], k in A[i]};
param z{i in numcons, j in cons[i]} = sum{k in events[j]} x[primitives[k,1], primitives[k,2], primitives[k,3]] * P[primitives[k,1], primitives[k,3], primitives[k,2], primitives[k,4]];
var zstar{i in numcons, j in cons[i]};


maximize ER: 
sum{i in 1..n, j in S[i], k in A[i]} Rcap[i,j,k]*x[i,j,k]*log(xstar[i,j,k]) + 
sum{l in numcons} crewardcap[l] * ( prod{m in cons[l]} z[l,m] ) * ( sum{m in cons[l]} log(zstar[l,m]) ) + 
sum{l in numcons} thetahat * sum{m in cons[l]} ((1 - z[l,m]) * log(1 - zstar[l,m])) + 
sum{l in numcons} thetahat * sum{m in cons[l]} z[l,m] * log(zstar[l,m]);

subject to Flow {i in 1..n, j in S[i]}:
sum{k in A[i]} xstar[i,j,k] - 
gamma * sum{l in S[i], m in A[i]} xstar[i,l,m]*P[i,m,l,j] == alpha[i,j];
subject to Positive {i in 1..n, j in S[i], k in A[i]}: xstar[i,j,k] >= 0.000001;

subject to defineZ{i in numcons, j in cons[i]}:
zstar[i,j] == sum{k in events[j]} xstar[primitives[k,1], primitives[k,2], primitives[k,3]] * P[primitives[k,1], primitives[k,3], primitives[k,2], primitives[k,4]];
