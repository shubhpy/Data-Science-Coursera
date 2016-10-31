[m n] = size(X);
X=[X zeros(m,n)];
for k=(2*n./2)+1:2*n
    for i=1:m
        X(i,k)=X(i,k-(2*n./2)).^2;
    end 
end
    